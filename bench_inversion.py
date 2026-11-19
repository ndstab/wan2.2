"""Feasibility benchmark for PreciseInv-style inversion on Wan2.2 TI2V-5B.

Measures the three numbers that gate the Tier-4 diffusion-inversion direction:

  1. Peak GPU memory for one forward+backward through the DiT at our operating
     point (704x1280, 61 frames). Answers "does this fit on one 80GB A100?".
  2. Wall time for forward, and for forward+backward.
  3. Iterations for the PreciseInv inner loop to converge on a single timestep.

This does NOT modify anything in wan/. It loads only the DiT (no VAE, no T5) and
drives it with synthetic latents -- we are measuring compute and memory, not
output quality, so real data is unnecessary.

Gradients flow ONLY to the noise tensor eps*. Every model parameter is frozen
with requires_grad_(False), which is what keeps this inside one card: we never
allocate the 5B-parameter gradient buffers that fine-tuning would need.

Usage:
    python bench_inversion.py --ckpt_dir /path/to/Wan2.2-TI2V-5B
"""

import argparse
import math
import time
import traceback

import torch
import torch.utils.checkpoint as ckpt

from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.modules.model import WanModel


def human_gb(n_bytes):
    return n_bytes / (1024 ** 3)


def enable_grad_checkpointing(model):
    """Wrap each DiT block in activation checkpointing.

    Trades compute for memory: block activations are discarded after the
    forward and recomputed during the backward, so only the 30 block
    boundaries stay resident instead of every intermediate. Monkeypatched
    here so nothing in wan/ is modified.
    """
    for blk in model.blocks:
        if getattr(blk, '_ckpt_wrapped', False):
            continue
        orig = blk.forward

        def make(fn):
            def wrapped(*args, **kwargs):
                return ckpt.checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return wrapped

        blk.forward = make(orig)
        blk._ckpt_wrapped = True
    return model


def free():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_inputs(cfg, model, device, frame_num, size):
    """Synthetic latents + text context matching the real inference shapes."""
    z_dim = 48  # vae z_dim for TI2V-5B
    f_lat = (frame_num - 1) // cfg.vae_stride[0] + 1
    h_lat = size[1] // cfg.vae_stride[1]
    w_lat = size[0] // cfg.vae_stride[2]
    latent_shape = (z_dim, f_lat, h_lat, w_lat)

    seq_len = int(math.ceil(
        (h_lat * w_lat) / (cfg.patch_size[1] * cfg.patch_size[2]) * f_lat))

    # Stand-in for the VAE-encoded clean reference latent.
    x0 = torch.randn(latent_shape, device=device, dtype=torch.float32)

    # Text context: [L, text_dim]. Random is fine -- we measure cost, not quality.
    context = [torch.randn(model.text_len, model.text_dim,
                           device=device, dtype=torch.float32)]

    return x0, context, seq_len, latent_shape


def euler_step(model, x_t, sigma_i, sigma_prev, t_val, context, seq_len,
               param_dtype, context_null=None, guide_scale=0.0):
    """One Euler step of the flow-matching ODE: x - (sigma_i - sigma_prev) * v.

    This is PreciseInv Algorithm 4's mu(x_t, t_i, t_{i-1}).

    With guide_scale > 0 this runs the two-pass classifier-free-guidance form
    used by our actual generation loop, so BOTH passes stay in the autograd
    graph and the backward traverses the DiT twice.
    """
    t = torch.tensor([t_val], device=x_t.device, dtype=torch.float32)
    xi = x_t.to(param_dtype)
    with torch.amp.autocast('cuda', dtype=param_dtype):
        v_cond = model([xi], t=t, context=context, seq_len=seq_len)[0]
        if guide_scale > 0.0:
            v_uncond = model([xi], t=t, context=context_null,
                             seq_len=seq_len)[0]
    if guide_scale > 0.0:
        v = v_uncond.float() + guide_scale * (v_cond.float() - v_uncond.float())
    else:
        v = v_cond.float()
    return x_t - (sigma_i - sigma_prev) * v


def measure(label, fn, warmup=False):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    if not warmup:
        print(f"  {label:<34} {dt:7.2f} s   peak {human_gb(peak):6.2f} GB")
    return out, dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--frame_num', type=int, default=61)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=704)
    ap.add_argument('--max_iters', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--cfg_scale', type=float, default=0.0,
                    help='0 = single pass; 5.0 = match our generation CFG')
    args = ap.parse_args()

    cfg = ti2v_5B
    device = torch.device('cuda')
    param_dtype = cfg.param_dtype
    size = (args.width, args.height)

    print("=" * 72)
    print("PreciseInv feasibility benchmark -- Wan2.2 TI2V-5B")
    print("=" * 72)
    print(f"  resolution      {args.width}x{args.height}, {args.frame_num} frames")
    print(f"  param_dtype     {param_dtype}")
    print(f"  CFG             {'OFF (1 pass/iter)' if args.cfg_scale <= 0 else f'ON, scale {args.cfg_scale} (2 passes/iter)'}")
    print(f"  GPU             {torch.cuda.get_device_name(0)}")
    print(f"  total VRAM      {human_gb(torch.cuda.get_device_properties(0).total_memory):.1f} GB")

    # ---- load DiT only -------------------------------------------------
    print("\n[1] Loading DiT (no VAE, no T5)...")
    t0 = time.time()
    model = WanModel.from_pretrained(args.ckpt_dir)
    model = model.eval().requires_grad_(False).to(device)
    if param_dtype is not None:
        model = model.to(param_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    loaded in {time.time() - t0:.1f} s   "
          f"{n_params / 1e9:.2f}B params   "
          f"weights {human_gb(n_params * 2):.1f} GB (bf16)")
    print(f"    all parameters frozen: "
          f"{not any(p.requires_grad for p in model.parameters())}")

    x0, context, seq_len, latent_shape = build_inputs(
        cfg, model, device, args.frame_num, size)
    context_null = [torch.randn_like(context[0])]
    cfg_kw = dict(context_null=context_null, guide_scale=args.cfg_scale)
    n_elem = int(torch.tensor(latent_shape).prod())
    print(f"    latent shape    {tuple(latent_shape)}  ({n_elem:,} elements)")
    print(f"    seq_len         {seq_len:,} tokens")

    # ---- 1. forward only ----------------------------------------------
    print("\n[2] Forward pass only (no grad)")
    sigma_i, sigma_prev = 0.5, 0.48
    eps = torch.randn_like(x0)
    x_t_nograd = (1 - sigma_i) * x0 + sigma_i * eps

    def fwd_only():
        with torch.no_grad():
            return euler_step(model, x_t_nograd, sigma_i, sigma_prev,
                              sigma_i * cfg.num_train_timesteps, context,
                              seq_len, param_dtype, **cfg_kw)

    measure("warmup (excluded)", fwd_only, warmup=True)
    _, fwd_dt, fwd_peak = measure("forward, no_grad", fwd_only)

    # ---- 2. forward + backward to eps* ---------------------------------
    eps_star = torch.randn_like(x0).requires_grad_(True)
    target = ((1 - sigma_prev) * x0 + sigma_prev * torch.randn_like(x0)).detach()

    def fwd_bwd():
        if eps_star.grad is not None:
            eps_star.grad = None
        x_t = (1 - sigma_i) * x0 + sigma_i * eps_star
        mu = euler_step(model, x_t, sigma_i, sigma_prev,
                        sigma_i * cfg.num_train_timesteps, context, seq_len,
                        param_dtype, **cfg_kw)
        loss = ((mu - target) ** 2).mean()
        loss.backward()
        return loss.item()

    print("\n[3] Forward + backward, NAIVE (no activation checkpointing)")
    naive_ok, fb_dt, fb_peak = True, None, None
    try:
        _, fb_dt, fb_peak = measure("forward + backward", fwd_bwd)
        print(f"    backward overhead     {fb_dt / max(fwd_dt, 1e-9):.2f}x forward")
    except torch.OutOfMemoryError as e:
        naive_ok = False
        msg = str(e).split('.')[0]
        print(f"  forward + backward                 OUT OF MEMORY  ({msg})")
        print("    -> naive backprop through all 30 blocks does NOT fit on this card")
    free()

    # ---- 3. same, with activation checkpointing ------------------------
    print("\n[4] Forward + backward, WITH activation checkpointing")
    enable_grad_checkpointing(model)
    measure("warmup (excluded)", fwd_bwd, warmup=True)
    free()
    _, ck_dt, ck_peak = measure("forward + backward (ckpt)", fwd_bwd)
    print(f"    slowdown vs forward   {ck_dt / max(fwd_dt, 1e-9):.2f}x")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"    headroom on this card ~{human_gb(total_vram - ck_peak):.1f} GB")
    fb_dt, fb_peak = ck_dt, ck_peak

    # ---- 3. inner-loop convergence -------------------------------------
    # PreciseInv Algorithm 4, lines 15-19, on a single timestep.
    # NOTE: the paper's eta thresholds are stated for SD-scale latents
    # (4x64x64 = 16k elements). Ours is ~170x larger, so we report BOTH the
    # mean-squared error and the total squared norm -- eta is scale-dependent
    # and their numbers do not transfer directly.
    print("\n[5] Inner-loop convergence, single timestep "
          f"(GD, lr={args.lr}, cap {args.max_iters} iters, ckpt enabled)")
    print(f"    {'iter':>5}  {'MSE':>12}  {'sum-sq':>14}  {'grad-norm':>11}  {'elapsed':>9}")

    eps_star = torch.randn_like(x0).requires_grad_(True)
    opt = torch.optim.Adam([eps_star], lr=args.lr)
    t_loop0 = time.time()
    history = []
    for it in range(args.max_iters):
        opt.zero_grad(set_to_none=True)
        x_t = (1 - sigma_i) * x0 + sigma_i * eps_star
        mu = euler_step(model, x_t, sigma_i, sigma_prev,
                        sigma_i * cfg.num_train_timesteps, context, seq_len,
                        param_dtype, **cfg_kw)
        diff = mu - target
        mse = (diff ** 2).mean()
        sumsq = (diff ** 2).sum()
        # Backprop the SUM of squares -- this is the paper's ||.||^2. Using the
        # mean divides the gradient by 2.7M elements and the step vanishes.
        sumsq.backward()
        gnorm = eps_star.grad.norm().item()
        opt.step()
        history.append((it, mse.item(), sumsq.item()))
        if it < 5 or it % 5 == 0 or it == args.max_iters - 1:
            print(f"    {it:>5}  {mse.item():>12.6e}  {sumsq.item():>14.6e}  "
                  f"{gnorm:>11.3e}  {time.time() - t_loop0:>8.1f}s")

    loop_dt = time.time() - t_loop0
    per_iter = loop_dt / args.max_iters

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  forward only                {fwd_dt:.2f} s   peak {human_gb(fwd_peak):.1f} GB")
    print(f"  fwd+bwd, naive              "
          f"{'OOM -- does not fit' if not naive_ok else f'{fb_dt:.2f} s   peak {human_gb(fb_peak):.1f} GB'}")
    print(f"  fwd+bwd, checkpointed       {ck_dt:.2f} s   peak {human_gb(ck_peak):.1f} GB "
          f"/ {human_gb(total_vram):.0f} GB")
    print(f"  measured per-iteration      {per_iter:.2f} s")
    print(f"  FITS ON ONE 80GB CARD:      "
          f"{'YES, with activation checkpointing' if ck_peak < total_vram else 'NO'}")
    first, last = history[0][1], history[-1][1]
    print(f"  MSE {first:.4e} -> {last:.4e} over {args.max_iters} iters "
          f"({first / max(last, 1e-30):.1f}x reduction)")
    print()
    print("  Extrapolated cost of a full reference inversion:")
    for T in (2, 5, 10, 50):
        for iters in (20, 50, 100):
            total = T * iters * per_iter
            print(f"    T={T:>2} steps x {iters:>3} iters = "
                  f"{total / 3600:6.2f} h")
        print()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # The 2026-05-17 lesson: never let a bare except swallow the message.
        print("\nFAILED:\n" + traceback.format_exc())
        raise
