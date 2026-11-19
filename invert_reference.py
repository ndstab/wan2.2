"""PreciseInv-style inversion of a reference video for Wan2.2 TI2V-5B.

Replaces the random re-noising at wan/textimage2video.py:452

    ref_noised = (1 - sigma) * ref_clean + sigma * torch.randn(...)
                                                   ^^^^^^^^^^^^^^^
with a *solved* noise term, one per denoising timestep:

    ref_noised = (1 - sigma) * ref_clean + sigma * eps_star[t]

This is PreciseInv (NeurIPS 2025) Algorithm 4 -- the rectified-flow / Euler
variant -- run on OUR 50-step schedule rather than the paper's T=2, so that the
solved sequence lines up exactly with the timesteps we inject at. Algorithm 4
returns a *sequence* of eps* (one per subproblem), which is precisely what the
injection pathway needs.

Solver note: the inner step mu() is a plain Euler step, not UniPC. UniPC keeps
a history of model outputs from noisier steps, which a bottom-up inversion has
not reached yet -- the dependency is circular. Generation is unaffected and
still runs UniPC; only this offline inversion uses Euler.

Deviation from Algorithm 4 (opt-in, on by default): eps* for subproblem t is
warm-started from the solution to t-1 rather than re-drawn from N(0,I).
Adjacent sigmas are close, so this converges far faster. The paper names "the
absence of an effective initialization manner" as the reason its SD3 (rectified
flow) numbers are slow, and lists better initialisation as future work.

Output: a single .pt holding eps* indexed by generation-order step index,
plus the sigma/timestep grid it was solved on, so the consumer can assert the
schedule matches.

Usage:
    python invert_reference.py \
        --ckpt_dir  /path/to/Wan2.2-TI2V-5B \
        --ref_video /path/to/sandstorm.mp4 \
        --out       ~/ref_inversion_sandstorm.pt
"""

import argparse
import os
import time
import traceback

import torch
import torch.utils.checkpoint as ckpt

from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def human_gb(n):
    return n / (1024 ** 3)


def enable_grad_checkpointing(model):
    """Activation checkpointing per DiT block.

    Measured on an A100 80GB at 704x1280/61f: naive backprop needs >79GB and
    OOMs; checkpointed it needs 19GB (25GB with CFG). Not optional.
    """
    for blk in model.blocks:
        if getattr(blk, '_ckpt_wrapped', False):
            continue
        orig = blk.forward

        def make(fn):
            def wrapped(*a, **kw):
                return ckpt.checkpoint(fn, *a, use_reentrant=False, **kw)
            return wrapped

        blk.forward = make(orig)
        blk._ckpt_wrapped = True


def load_reference_latent(vae, ref_video_path, frame_num, size, device):
    """Encode the reference clip exactly as WanTI2V.t2v() does.

    Mirrors wan/textimage2video.py:335-349 -- uniform frame sampling, [-1,1]
    normalisation, bilinear resize to the content resolution so the latent grid
    (and therefore RoPE) matches the content stream token-for-token.
    """
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(ref_video_path, ctx=decord.cpu(0))
    total = len(vr)
    idx = torch.linspace(0, total - 1, steps=frame_num).round().long()
    idx = idx.clamp(0, total - 1)
    frames = vr.get_batch(idx.tolist())                    # [F,H,W,C] uint8
    frames = frames.permute(3, 0, 1, 2).float() / 255.0    # [C,F,H,W]
    frames = frames * 2.0 - 1.0
    frames = torch.nn.functional.interpolate(
        frames.permute(1, 0, 2, 3),
        size=(size[1], size[0]), mode='bilinear', align_corners=False,
    ).permute(1, 0, 2, 3).contiguous()
    frames = frames.to(device=device, dtype=torch.float32)
    return vae.encode([frames])[0].to(dtype=torch.float32)


def velocity(model, x, sigma, context, seq_len, param_dtype,
             num_train_timesteps, context_null=None, guide_scale=0.0):
    """v_theta(x_sigma, t). One DiT forward, or two under CFG."""
    t = torch.tensor([sigma * num_train_timesteps],
                     device=x.device, dtype=torch.float32)
    xi = x.to(param_dtype)
    with torch.amp.autocast('cuda', dtype=param_dtype):
        v_cond = model([xi], t=t, context=context, seq_len=seq_len)[0]
        if guide_scale > 0.0:
            v_unc = model([xi], t=t, context=context_null, seq_len=seq_len)[0]
    if guide_scale > 0.0:
        return v_unc.float() + guide_scale * (v_cond.float() - v_unc.float())
    return v_cond.float()


@torch.no_grad()
def reconstruct(model, x_start, sigmas_gen, context, seq_len, param_dtype,
                num_train_timesteps, context_null, guide_scale):
    """Run the full Euler denoising chain from the noisiest sigma down to 0.

    This is the honest test of an inversion: does the solved latent actually
    regenerate the reference? Comparing this against the same chain started
    from a RANDOM eps is the direct question -- is a solved eps* better than
    the torch.randn we use today?
    """
    x = x_start
    n = len(sigmas_gen)
    for i in range(n):
        s_hi = sigmas_gen[i]
        s_lo = sigmas_gen[i + 1] if i + 1 < n else 0.0
        v = velocity(model, x, s_hi, context, seq_len, param_dtype,
                     num_train_timesteps, context_null, guide_scale)
        x = x - (s_hi - s_lo) * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--ref_video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt', default='',
                    help='conditioning for the inversion; empty = null prompt')
    ap.add_argument('--frame_num', type=int, default=61)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=704)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--shift', type=float, default=5.0)
    ap.add_argument('--eta', type=float, default=1e-4,
                    help='convergence threshold on per-element MSE')
    ap.add_argument('--min_iters', type=int, default=3,
                    help='gradient steps taken before eta may stop a subproblem')
    ap.add_argument('--max_iters', type=int, default=80)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--cfg_scale', type=float, default=0.0)
    ap.add_argument('--cold_start', action='store_true',
                    help='re-draw eps* per subproblem, as written in Alg 4')
    args = ap.parse_args()

    cfg = ti2v_5B
    device = torch.device('cuda')
    param_dtype = cfg.param_dtype
    size = (args.width, args.height)
    out_path = os.path.expanduser(args.out)

    print("=" * 74)
    print("PreciseInv reference inversion -- Wan2.2 TI2V-5B")
    print("=" * 74)
    print(f"  ref video     {args.ref_video}")
    print(f"  resolution    {args.width}x{args.height}, {args.frame_num} frames")
    print(f"  schedule      {args.sample_steps} steps, shift {args.shift}")
    print(f"  eta           {args.eta}   min_iters {args.min_iters}   "
          f"max_iters {args.max_iters}   lr {args.lr}")
    print(f"  CFG           {'off' if args.cfg_scale <= 0 else args.cfg_scale}")
    print(f"  init          {'cold (Alg 4 as written)' if args.cold_start else 'warm-start from previous subproblem'}")
    print(f"  out           {out_path}")

    # ---- reference latent (VAE, then freed) ----------------------------
    print("\n[1] Encoding reference video...")
    vae = Wan2_2_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
                     device=device)
    x0 = load_reference_latent(vae, args.ref_video, args.frame_num, size, device)
    print(f"    ref latent    {tuple(x0.shape)}")
    del vae
    torch.cuda.empty_cache()

    # ---- text conditioning ---------------------------------------------
    # T5 is only needed for a non-empty prompt. Empty prompt -> zero context,
    # which is what the null branch uses anyway.
    print("\n[2] Building text context...")
    if args.prompt:
        from wan.modules.t5 import T5EncoderModel
        t5 = T5EncoderModel(
            text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
            checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer))
        context = t5([args.prompt], device)
        context_null = t5([cfg.sample_neg_prompt], device)
        del t5
        torch.cuda.empty_cache()
        print(f"    prompt        {args.prompt!r}")
    else:
        zeros = torch.zeros(cfg.text_len, 4096, device=device,
                            dtype=torch.float32)
        context, context_null = [zeros], [zeros]
        print("    prompt        (empty / null conditioning)")

    # ---- DiT -------------------------------------------------------------
    print("\n[3] Loading DiT...")
    model = WanModel.from_pretrained(args.ckpt_dir)
    model = model.eval().requires_grad_(False).to(device).to(param_dtype)
    enable_grad_checkpointing(model)
    print(f"    {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
          f"frozen, activation checkpointing on")

    # ---- schedule: identical to generation -------------------------------
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(args.sample_steps, device=device, shift=args.shift)
    timesteps = sched.timesteps                      # descending: noisy -> clean
    sigmas_gen = (timesteps.float() / cfg.num_train_timesteps).tolist()

    # Bottom-up inversion walks clean -> noisy, i.e. the reverse.
    order_asc = list(range(len(sigmas_gen) - 1, -1, -1))
    seq_len = int(x0.shape[1] * (x0.shape[2] // cfg.patch_size[1]) *
                  (x0.shape[3] // cfg.patch_size[2]))
    print(f"    seq_len       {seq_len:,} tokens")
    print(f"    sigma range   {min(sigmas_gen):.4f} .. {max(sigmas_gen):.4f}")

    # ---- Algorithm 4 -----------------------------------------------------
    print(f"\n[4] Solving {len(order_asc)} subproblems (bottom-up)\n")
    print(f"    {'gen_i':>6} {'sigma':>8} {'iters':>6} {'final MSE':>12} "
          f"{'converged':>10} {'secs':>7}")

    eps_store, meta = {}, []
    x_prev = x0.clone()                    # P(1) target is the clean latent
    eps_prev = None
    t_start = time.time()

    for k, gen_i in enumerate(order_asc):
        sigma_i = sigmas_gen[gen_i]
        sigma_prev = sigmas_gen[order_asc[k - 1]] if k > 0 else 0.0

        if eps_prev is None or args.cold_start:
            eps_star = torch.randn_like(x0)
        else:
            eps_star = eps_prev.clone()
        eps_star.requires_grad_(True)
        opt = torch.optim.Adam([eps_star], lr=args.lr)

        t0, n_it, mse_val, converged = time.time(), 0, float('inf'), False
        for n_it in range(1, args.max_iters + 1):
            opt.zero_grad(set_to_none=True)
            x_t = (1 - sigma_i) * x0 + sigma_i * eps_star
            v = velocity(model, x_t, sigma_i, context, seq_len, param_dtype,
                         cfg.num_train_timesteps, context_null, args.cfg_scale)
            mu = x_t - (sigma_i - sigma_prev) * v
            diff = mu - x_prev
            mse_val = (diff ** 2).mean().item()
            # min_iters guards a degenerate case found in the 3-step smoke run:
            # with warm-starting, a loose eta lets a subproblem "converge" at
            # iteration 1 having taken NO gradient step, so eps* is copied
            # unchanged from the previous subproblem. Chain that up the whole
            # schedule and the "inversion" is one random draw reused.
            if mse_val < args.eta and n_it > args.min_iters:
                converged = True
                break
            # Backprop the SUM of squares: this is the paper's ||.||^2. Using
            # the mean divides the gradient by ~2.7M and the step vanishes.
            (diff ** 2).sum().backward()
            opt.step()

        eps_star = eps_star.detach()
        with torch.no_grad():
            x_prev = (1 - sigma_i) * x0 + sigma_i * eps_star
        eps_prev = eps_star
        eps_store[gen_i] = eps_star.to(torch.float16).cpu()
        dt = time.time() - t0
        meta.append(dict(gen_i=gen_i, sigma=sigma_i, iters=n_it,
                         mse=mse_val, converged=converged, secs=dt))
        print(f"    {gen_i:>6} {sigma_i:>8.4f} {n_it:>6} {mse_val:>12.4e} "
              f"{str(converged):>10} {dt:>7.1f}")

        # Save after every subproblem -- this is a multi-hour job and a crash
        # at step 47 must not cost the previous 46.
        torch.save(dict(
            eps=eps_store,
            sigmas_gen=sigmas_gen,
            timesteps=timesteps.cpu(),
            meta=meta,
            config=dict(frame_num=args.frame_num, width=args.width,
                        height=args.height, sample_steps=args.sample_steps,
                        shift=args.shift, eta=args.eta, lr=args.lr,
                        cfg_scale=args.cfg_scale, prompt=args.prompt,
                        cold_start=args.cold_start,
                        ref_video=os.path.abspath(args.ref_video)),
            latent_shape=tuple(x0.shape),
        ), out_path)

    total = time.time() - t_start
    n_conv = sum(m['converged'] for m in meta)
    print("\n" + "=" * 74)
    print(f"  solved {len(meta)} subproblems in {total / 3600:.2f} h")
    print(f"  converged below eta={args.eta}: {n_conv}/{len(meta)}")
    print(f"  mean iters/step  {sum(m['iters'] for m in meta) / len(meta):.1f}")
    print(f"  final MSE  min {min(m['mse'] for m in meta):.3e}  "
          f"max {max(m['mse'] for m in meta):.3e}")
    print(f"  peak GPU   {human_gb(torch.cuda.max_memory_allocated()):.1f} GB")

    # ---- validation: does it actually reconstruct? ----------------------
    print("\n[5] Reconstruction check (full Euler chain, inverted vs random)")
    s_max = sigmas_gen[0]
    vkw = dict(context=context, seq_len=seq_len, param_dtype=param_dtype,
               num_train_timesteps=cfg.num_train_timesteps,
               context_null=context_null, guide_scale=args.cfg_scale)

    x_inv = (1 - s_max) * x0 + s_max * eps_store[0].to(device).float()
    x_rnd = (1 - s_max) * x0 + s_max * torch.randn_like(x0)

    def report(name, x_start):
        rec = reconstruct(model, x_start, sigmas_gen, **vkw)
        mse = ((rec - x0) ** 2).mean().item()
        rel = ((rec - x0).norm() / x0.norm()).item()
        print(f"    {name:<24} MSE {mse:.5e}   rel-L2 {rel:.4f}")
        return mse, rel

    inv_mse, inv_rel = report("inverted eps*", x_inv)
    rnd_mse, rnd_rel = report("random eps (current)", x_rnd)
    print(f"\n    improvement: MSE {rnd_mse / max(inv_mse, 1e-30):.2f}x better, "
          f"rel-L2 {rnd_rel:.4f} -> {inv_rel:.4f}")
    if inv_mse >= rnd_mse:
        print("    WARNING: inversion is NOT better than random. Do not build "
              "on this cache until understood.")
    print(f"  written to {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.0f} MB)")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\nFAILED:\n" + traceback.format_exc())
        raise
