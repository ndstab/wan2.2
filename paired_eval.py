"""T5.5 -- paired evaluation of weather steering against ground truth.

For a held-out (clear, weather) pair from a WeatherWeaver shard:

  1. noise the CLEAR latent to sigma_start (SDEdit-style; same eps for every arm),
  2. denoise to sigma=0 with the 50-step UniPC schedule, null conditioning, no
     CFG -- the same conditioning the steering directions were measured under,
  3. arms:
       base      no steering            (how far is "just re-render the clear video")
       a=<x>     steering at alpha x    (the method)
       oracle    same, but start from the WEATHER latent, no steering
                 (the best this SDEdit protocol can do; the error floor)
  4. compare each output with the real weather render of the same camera path.

Directions are leave-one-shard-out: the shard being evaluated never contributes
to its own steering direction. --direction global averages every other pair;
--direction knn averages the k pairs whose weather vector is closest to the
target's (a weather-conditioned direction, still a plain mean of activations).

Metrics (latent space, always):
  frac  <x_hat - x_base, D> / |D|^2   fraction of the true weather difference
                                      D = x0_weather - x0_clear realised
  cos   cos(x_hat - x_base, D)        does the change point the right way, per token
  err   |x_hat - x0_weather| / |D|    distance to the real weather render
Pixel space (--pixel): PSNR vs the VAE reconstruction of the weather render
(so VAE loss is not charged to the method), and LPIPS if the lpips package and
its weights are available offline.

No gradients, no optimiser, no weight changes.

Cost: each arm is ~(50 - k) forward passes, k = the step at sigma_start
(~40 passes at sigma_start 0.95). Budget roughly 1 min per arm per pair.
"""
import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict

import torch

from paired_atlas import load_video_latent, shard_pairs
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.steering import (WeatherSteerer, build_directions, load_records,
                                nearest_records)


def latent_metrics(x_hat, x_base, x0c, x0w):
    D = (x0w - x0c).flatten()
    a = (x_hat - x_base).flatten()
    dd = D.dot(D).clamp_min(1e-12)
    return dict(
        frac=(a.dot(D) / dd).item(),
        cos=torch.nn.functional.cosine_similarity(a, D, dim=0).item()
        if a.norm() > 0 else 0.0,
        err=((x_hat - x0w).norm() / dd.sqrt()).item(),
    )


def psnr(a, b):
    """a, b in [-1, 1]."""
    mse = ((a.float() - b.float()) / 2).pow(2).mean().clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).item()


def make_lpips(device):
    try:
        import lpips
        return lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    except Exception as e:      # no package, or weights not cached offline
        print(f"LPIPS unavailable ({type(e).__name__}: {e}); skipping", flush=True)
        return None


def lpips_video(fn, a, b, stride=6):
    """Mean LPIPS over every stride-th frame. a, b: [C, F, H, W] in [-1, 1]."""
    fa = a[:, ::stride].permute(1, 0, 2, 3)
    fb = b[:, ::stride].permute(1, 0, 2, 3)
    return fn(fa.float(), fb.float()).mean().item()


def panel(v, h=176, w=320):
    return torch.nn.functional.interpolate(
        v.permute(1, 0, 2, 3).float(), size=(h, w), mode='bilinear',
        align_corners=False).permute(1, 0, 2, 3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--shards', required=True, help='glob of eval .tar shards')
    ap.add_argument('--records', required=True, help='paired_atlas.py --out dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--direction', choices=['global', 'knn'], default='global')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--alphas', default='1,2,4')
    ap.add_argument('--steer_blocks', default='19,20,21,22')
    ap.add_argument('--steer_sigma', default='0.83,0.95')
    ap.add_argument('--norm_preserve', action='store_true')
    ap.add_argument('--start_sigmas', default='0.95',
                    help='SDEdit start noise levels (comma-separated)')
    ap.add_argument('--no_oracle', action='store_true')
    ap.add_argument('--pixel', action='store_true', help='decode + PSNR/LPIPS')
    ap.add_argument('--save_videos', type=int, default=4,
                    help='write comparison mp4s for the first N pairs')
    ap.add_argument('--frame_num', type=int, default=61)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--shift', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_pairs', type=int, default=0)
    args = ap.parse_args()

    import glob
    from wan.utils.utils import save_video

    cfg = ti2v_5B
    device = torch.device('cuda')
    size = (1280, 704)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)

    vae = Wan2_2_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
                     device=device)
    model = WanModel.from_pretrained(args.ckpt_dir)
    model = model.eval().requires_grad_(False).to(device).to(cfg.param_dtype)
    context = [torch.zeros(cfg.text_len, 4096, device=device, dtype=torch.float32)]
    lp = make_lpips(device) if args.pixel else None

    def make_sched():
        s = FlowUniPCMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps, shift=1,
            use_dynamic_shifting=False)
        s.set_timesteps(args.sample_steps, device=device, shift=args.shift)
        return s

    timesteps = make_sched().timesteps
    sigmas_all = (timesteps.float() / cfg.num_train_timesteps).tolist()
    starts = []
    for s in (float(x) for x in args.start_sigmas.split(',')):
        k = next(i for i, v in enumerate(sigmas_all) if v <= s)
        starts.append((k, sigmas_all[k]))
    print("start steps: " + ", ".join(f"k={k} sigma={s:.3f}" for k, s in starts),
          flush=True)

    alphas = [float(a) for a in args.alphas.split(',')]
    blocks = [int(b) for b in args.steer_blocks.split(',')]
    window = [float(s) for s in args.steer_sigma.split(',')]
    records = load_records(args.records)
    print(f"{len(records)} direction records", flush=True)

    seq_len = None

    def forward(x, t):
        with torch.amp.autocast('cuda', dtype=cfg.param_dtype):
            v = model([x.to(cfg.param_dtype)], t=t.reshape(1).float(),
                      context=context, seq_len=seq_len)[0]
        return v.float()

    def denoise(x0, eps, k, steerer):
        sched = make_sched()
        sig = sigmas_all[k]
        x = (1 - sig) * x0 + sig * eps
        if steerer is not None:
            steerer.n_applied = 0
            steerer.attach(model)
        try:
            for t in sched.timesteps[k:]:
                v = forward(x, t)
                x = sched.step(v.unsqueeze(0), t, x.unsqueeze(0),
                               return_dict=False)[0].squeeze(0)
        finally:
            if steerer is not None:
                steerer.detach()
        return x

    csv_path = os.path.join(args.out, 'metrics.csv')
    fields = ['shard', 'variant', 'params', 'start_sigma', 'arm', 'alpha',
              'n_dir_pairs', 'n_applied', 'frac', 'cos', 'err',
              'psnr_weather', 'psnr_clear', 'lpips_weather', 'lpips_clear']
    new = not os.path.exists(csv_path)
    fcsv = open(csv_path, 'a', newline='')
    wr = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        wr.writeheader()
    agg = defaultdict(list)

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
        dec_c = vae.decode([x0c])[0] if args.pixel or n_pairs < args.save_videos else None

        for _, _, vid, params in pairs:
            if args.max_pairs and n_pairs >= args.max_pairs:
                break
            t0 = time.time()
            x0w = lat[vid]
            if args.direction == 'global':
                dsel = [r for r in records if r['shard'] != shard]
            else:
                dsel = nearest_records(records, params, args.k, exclude_shards=[shard])
            dirs = build_directions(dsel)
            pstr = ",".join(f"{k}{v:g}" for k, v in params.items() if v)
            want_dec = args.pixel or n_pairs < args.save_videos
            dec_w = vae.decode([x0w])[0] if want_dec else None

            for k, s0 in starts:
                g = torch.Generator(device=device).manual_seed(
                    args.seed * 1000 + n_pairs)
                eps = torch.randn(x0c.shape, generator=g, device=device)
                arms = [('base', 0.0, x0c, None)]
                for a in alphas:
                    arms.append((f'a={a:g}', a, x0c, WeatherSteerer(
                        dirs, alpha=a, blocks=blocks, sigma_window=window,
                        norm_preserve=args.norm_preserve)))
                if not args.no_oracle:
                    arms.append(('oracle', 0.0, x0w, None))

                outs, x_base, vids = {}, None, []
                for name, a, start, st in arms:
                    x_hat = denoise(start, eps, k, st)
                    if name == 'base':
                        x_base = x_hat
                    m = latent_metrics(x_hat, x_base, x0c, x0w)
                    row = dict(shard=shard, variant=vid, params=pstr,
                               start_sigma=f"{s0:.3f}", arm=name, alpha=a,
                               n_dir_pairs=dirs['n_pairs'],
                               n_applied=st.n_applied if st else 0, **m)
                    if want_dec:
                        dec = vae.decode([x_hat])[0]
                        vids.append(dec)
                        if args.pixel:
                            row['psnr_weather'] = psnr(dec, dec_w)
                            row['psnr_clear'] = psnr(dec, dec_c)
                            if lp is not None:
                                row['lpips_weather'] = lpips_video(lp, dec, dec_w)
                                row['lpips_clear'] = lpips_video(lp, dec, dec_c)
                    if st is not None and st.n_applied == 0:
                        print(f"WARNING {name}: steering never fired", flush=True)
                    wr.writerow(row)
                    fcsv.flush()
                    for key in ('frac', 'cos', 'err', 'psnr_weather', 'lpips_weather'):
                        if key in row:
                            agg[(f"{s0:.3f}", name, key)].append(row[key])
                    outs[name] = m

                if n_pairs < args.save_videos and vids:
                    # clear | real weather | base | steered... | oracle
                    grid = torch.cat([panel(dec_c), panel(dec_w)] +
                                     [panel(v) for v in vids], dim=3)
                    save_video(tensor=grid[None], save_file=os.path.join(
                        args.out, f"{shard}_{vid}_s{s0:.2f}.mp4"),
                        fps=cfg.sample_fps, nrow=1, normalize=True,
                        value_range=(-1, 1))
                summ = "  ".join(f"{n}: frac {o['frac']:.2f} cos {o['cos']:.2f} "
                                 f"err {o['err']:.2f}" for n, o in outs.items())
                print(f"[{n_pairs + 1}] {shard}_{vid} ({pstr}) s0={s0:.2f}  {summ}",
                      flush=True)
            print(f"    {time.time() - t0:.0f}s", flush=True)
            n_pairs += 1
        del lat
        torch.cuda.empty_cache()
        if args.max_pairs and n_pairs >= args.max_pairs:
            break
    fcsv.close()

    summary = {f"{s}|{arm}|{key}": dict(
        mean=sum(v) / len(v), n=len(v)) for (s, arm, key), v in agg.items()}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print("\nmean over pairs:")
    for key in sorted(summary):
        print(f"  {key:40s} {summary[key]['mean']:.4f}  (n={summary[key]['n']})")
    print("PAIRED_EVAL_DONE", flush=True)


if __name__ == '__main__':
    main()
