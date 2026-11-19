"""T5.3b -- paired deterministic inversion: where along the trajectory does weather live?

paired_atlas.py noises both members of a pair with the SAME random eps (forward
noising). This script instead runs the deterministic ODE inversion of each
member separately -- each video is walked from sigma=0 up to sigma_max along
the 50-step schedule with explicit Euler steps of the model's own velocity
(the flow-matching analogue of DDIM inversion), null conditioning, no CFG:

    x(s_{i+1}) = x(s_i) + (s_{i+1} - s_i) * v(x(s_i), s_i)

Both inversions are run in lockstep, so at every step we compare the two
latents AND the two models' internal states at all 30 blocks:

  lat_ratio  |x_w(s) - x_c(s)| / |x_w(0) - x_c(0)|
             how much of the weather difference is still present in the
             inverted latent at noise level s. Where it collapses, the model's
             trajectory no longer distinguishes the two videos.
  lat_cos    cos(x_w(s) - x_c(s), x_w(0) - x_c(0))  -- same direction or not
  rel[b]     |h_w[b] - h_c[b]| / |h_c[b]|  per block, along the inversion
  dmean[b]   token-mean h_w[b] - h_c[b]  (comparable with the atlas directions)

At the end: cosine between the two inverted noises. If clear and weather invert
to nearly the same noise, weather is something the trajectory adds late; if
not, the noise itself encodes it.

--recon re-generates each video from its inverted noise and reports the
reconstruction error, so we know how far to trust the inversion (Euler
inversion is approximate; see the T4 changelog for why that matters).

No gradients, no optimiser, no weight changes. Cost: 2 x sample_steps forward
passes per pair (~2.5 min), doubled with --recon.
"""
import argparse
import glob
import os
import tempfile
import time

import torch

from paired_atlas import load_video_latent, shard_pairs
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame_num', type=int, default=61)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--shift', type=float, default=5.0)
    ap.add_argument('--fp_iters', type=int, default=0,
                    help='fixed-point refinements per step: solve '
                         'x_{i+1} = x_i + ds * v(x_{i+1}, s_{i+1}), the exact '
                         'inverse of the Euler sampler used by --recon '
                         '(0 = plain explicit Euler inversion)')
    ap.add_argument('--probe_every', type=int, default=5,
                    help='record block stats every N inversion steps (+ last)')
    ap.add_argument('--recon', action='store_true')
    ap.add_argument('--max_pairs', type=int, default=0)
    args = ap.parse_args()

    cfg = ti2v_5B
    device = torch.device('cuda')
    size = (1280, 704)
    os.makedirs(args.out, exist_ok=True)

    vae = Wan2_2_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
                     device=device)
    model = WanModel.from_pretrained(args.ckpt_dir)
    model = model.eval().requires_grad_(False).to(device).to(cfg.param_dtype)
    n_blocks = len(model.blocks)
    context = [torch.zeros(cfg.text_len, 4096, device=device, dtype=torch.float32)]

    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1,
        use_dynamic_shifting=False)
    sched.set_timesteps(args.sample_steps, device=device, shift=args.shift)
    gen_sigmas = (sched.timesteps.float() / cfg.num_train_timesteps).tolist()
    # Inversion grid: 0 -> smallest sigma -> ... -> sigma_max (ascending).
    grid = [0.0] + gen_sigmas[::-1]
    n_steps = len(grid) - 1
    probe = sorted(set(range(0, n_steps, args.probe_every)) | {n_steps - 1})

    state = {'phase': None, 'store': [None] * n_blocks, 'stats': [None] * n_blocks}

    def make_hook(b):
        def hook(_m, _inp, out):
            h = out[0] if out.dim() == 3 else out
            if state['phase'] == 'store':
                state['store'][b] = h.detach().clone()
            elif state['phase'] == 'compare':
                hc = state['store'][b].float()
                d = h.float() - hc
                state['stats'][b] = dict(rel=(d.norm() / hc.norm()).item(),
                                         dmean=d.mean(0).half().cpu())
        return hook

    handles = [blk.register_forward_hook(make_hook(b))
               for b, blk in enumerate(model.blocks)]
    seq_len = None

    def velocity(x, sigma):
        # The model is never evaluated at sigma=0; the first inversion step uses
        # the smallest scheduled sigma, the usual DDIM-inversion convention.
        s = max(sigma, grid[1])
        t = torch.tensor([s * cfg.num_train_timesteps], device=device)
        with torch.amp.autocast('cuda', dtype=cfg.param_dtype):
            v = model([x.to(cfg.param_dtype)], t=t, context=context,
                      seq_len=seq_len)[0]
        return v.float()

    # regenerate() is the Euler sampler x(s_{i-1}) = x(s_i) + (s_{i-1} - s_i) v(x(s_i), s_i);
    # with --fp_iters the inversion above is its fixed-point inverse.
    def regenerate(x_T):
        x = x_T
        state['phase'] = None
        for i in range(n_steps, 0, -1):          # sigma_max -> 0, Euler
            x = x + (grid[i - 1] - grid[i]) * velocity(x, grid[i])
        return x

    n_pairs = 0
    for tar in sorted(glob.glob(args.shards)):
        shard = os.path.basename(tar)[:-4]
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', '/tmp')) as td:
            pairs = list(shard_pairs(tar, td))
            lat = {'clear': load_video_latent(vae, pairs[0][0], args.frame_num,
                                              size, device)}
            for _, w_mp4, vid, _ in pairs:
                lat[vid] = load_video_latent(vae, w_mp4, args.frame_num, size, device)
        x0c = lat['clear']
        T, Hl, Wl = x0c.shape[1:]
        seq_len = T * (Hl // cfg.patch_size[1]) * (Wl // cfg.patch_size[2])

        for _, _, vid, params in pairs:
            if args.max_pairs and n_pairs >= args.max_pairs:
                break
            out_path = os.path.join(args.out, f"{shard}_{vid}.pt")
            if os.path.exists(out_path):
                n_pairs += 1
                continue
            t0 = time.time()
            x0w = lat[vid]
            D0 = x0w - x0c
            nD0 = D0.norm()
            xc, xw = x0c.clone(), x0w.clone()
            rec = dict(shard=shard, variant=vid, params=params, grid=grid,
                       fp_iters=args.fp_iters,
                       probe=probe, latent_diff=nD0.item(),
                       lat_ratio=[], lat_cos=[], blocks={})
            for i in range(n_steps):
                is_probe = i in probe
                state['phase'] = 'store' if is_probe else None
                vc = velocity(xc, grid[i])
                state['phase'] = 'compare' if is_probe else None
                vw = velocity(xw, grid[i])
                state['phase'] = None
                if is_probe:
                    for b in range(n_blocks):
                        rec['blocks'][(b, i)] = state['stats'][b]
                ds = grid[i + 1] - grid[i]
                xc_n, xw_n = xc + ds * vc, xw + ds * vw
                for _ in range(args.fp_iters):
                    xc_n = xc + ds * velocity(xc_n, grid[i + 1])
                    xw_n = xw + ds * velocity(xw_n, grid[i + 1])
                xc, xw = xc_n, xw_n
                D = xw - xc
                rec['lat_ratio'].append((D.norm() / nD0).item())
                rec['lat_cos'].append(torch.nn.functional.cosine_similarity(
                    D.flatten(), D0.flatten(), dim=0).item())
            rec['noise_cos'] = torch.nn.functional.cosine_similarity(
                xc.flatten(), xw.flatten(), dim=0).item()
            rec['noise_std'] = (xc.std().item(), xw.std().item())
            if args.recon:
                rc, rw = regenerate(xc), regenerate(xw)
                rec['recon_rel'] = ((rc - x0c).norm() / x0c.norm()).item(), \
                                   ((rw - x0w).norm() / x0w.norm()).item()
                # Does the regenerated pair keep the weather difference?
                rec['recon_diff_ratio'] = ((rw - rc).norm() / nD0).item()
            torch.save(rec, out_path)
            # Where along the inversion does half the weather difference remain?
            half = next((grid[i + 1] for i, r in enumerate(rec['lat_ratio'])
                         if r < 0.5), None)
            pstr = ",".join(f"{k}{v:g}" for k, v in params.items() if v)
            print(f"[{n_pairs + 1}] {shard}_{vid} ({pstr})  "
                  f"ratio@end {rec['lat_ratio'][-1]:.3f}  "
                  f"half-life sigma {half if half is None else round(half, 3)}  "
                  f"noise_cos {rec['noise_cos']:.3f}"
                  + (f"  recon {rec['recon_rel'][0]:.3f}/{rec['recon_rel'][1]:.3f}"
                     if args.recon else "")
                  + f"  {time.time() - t0:.0f}s", flush=True)
            n_pairs += 1
        state['store'] = [None] * n_blocks
        del lat
        torch.cuda.empty_cache()
        if args.max_pairs and n_pairs >= args.max_pairs:
            break

    for h in handles:
        h.remove()
    print("PAIRED_INVERSION_DONE", flush=True)


if __name__ == '__main__':
    main()
