"""T5.5 (v2) -- paired evaluation of weather steering on an exactly inverted clear video.

Why v2: the SDEdit protocol in paired_eval.py (noise the clear latent to
sigma 0.95, denoise with a null prompt and no CFG) destroys the scene even
with no steering at all -- the unsteered arm is a featureless blur, so it
cannot measure what steering adds. Here the clear video is instead inverted
with the fixed-point Euler inversion of paired_inversion.py (--fp_iters 3
reconstructs at ~4% rel-L2), and regenerated from that noise:

  base      regenerate, no steering        (~ the clear video itself)
  a=<x>|w   regenerate, steering at alpha x inside sigma window w
  target    the real weather render of the same camera path

One inversion per shard serves all of that shard's weather variants.
Directions are leave-one-shard-out (global mean, or the k pairs nearest in
weather-parameter space). Latent metrics (frac, cos, err) as in paired_eval.py;
--pixel adds PSNR vs the VAE reconstruction of the weather / clear render.

No gradients, no optimiser, no weight changes.
"""
import argparse
import csv
import glob
import json
import os
import tempfile
import time
from collections import defaultdict

import torch

from paired_atlas import load_video_latent, shard_pairs
from paired_eval import latent_metrics, panel, psnr
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.steering import (WeatherSteerer, build_directions, load_records,
                                nearest_records)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--records', required=True, help='paired_atlas.py --out dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--direction', choices=['global', 'knn'], default='global')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--alphas', default='1,2,4')
    ap.add_argument('--windows', default='0.83,0.95;0.55,0.92',
                    help="';'-separated 'lo,hi' sigma windows")
    ap.add_argument('--steer_blocks', default='19,20,21,22')
    ap.add_argument('--norm_preserve', action='store_true')
    ap.add_argument('--fp_iters', type=int, default=3)
    ap.add_argument('--pixel', action='store_true')
    ap.add_argument('--save_videos', type=int, default=4)
    ap.add_argument('--frame_num', type=int, default=61)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--shift', type=float, default=5.0)
    args = ap.parse_args()

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

    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1,
        use_dynamic_shifting=False)
    sched.set_timesteps(args.sample_steps, device=device, shift=args.shift)
    grid = [0.0] + (sched.timesteps.float() / cfg.num_train_timesteps).tolist()[::-1]
    n_steps = len(grid) - 1
    seq_len = None

    def velocity(x, sigma):
        s = max(sigma, grid[1])
        t = torch.tensor([s * cfg.num_train_timesteps], device=device)
        with torch.amp.autocast('cuda', dtype=cfg.param_dtype):
            v = model([x.to(cfg.param_dtype)], t=t, context=context,
                      seq_len=seq_len)[0]
        return v.float()

    def invert(x0):
        x = x0.clone()
        for i in range(n_steps):
            ds = grid[i + 1] - grid[i]
            xn = x + ds * velocity(x, grid[i])
            for _ in range(args.fp_iters):
                xn = x + ds * velocity(xn, grid[i + 1])
            x = xn
        return x

    def regenerate(x_T, steerer=None):
        if steerer is not None:
            steerer.n_applied = 0
            steerer.attach(model)
        try:
            x = x_T
            for i in range(n_steps, 0, -1):
                x = x + (grid[i - 1] - grid[i]) * velocity(x, grid[i])
        finally:
            if steerer is not None:
                steerer.detach()
        return x

    alphas = [float(a) for a in args.alphas.split(',')]
    windows = [tuple(float(v) for v in w.split(',')) for w in args.windows.split(';')]
    blocks = [int(b) for b in args.steer_blocks.split(',')]
    records = load_records(args.records)

    csv_path = os.path.join(args.out, 'metrics.csv')
    fields = ['shard', 'variant', 'params', 'arm', 'alpha', 'window', 'n_dir_pairs',
              'n_applied', 'frac', 'cos', 'err', 'psnr_weather', 'psnr_clear']
    new = not os.path.exists(csv_path)
    fcsv = open(csv_path, 'a', newline='')
    wr = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        wr.writeheader()
    agg = defaultdict(list)
    n_saved = 0

    for tar in sorted(glob.glob(args.shards)):
        shard = os.path.basename(tar)[:-4]
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', '/tmp')) as td:
            pairs = list(shard_pairs(tar, td))
            x0c = load_video_latent(vae, pairs[0][0], args.frame_num, size, device)
            lat = {vid: load_video_latent(vae, w, args.frame_num, size, device)
                   for _, w, vid, _ in pairs}
        T, Hl, Wl = x0c.shape[1:]
        seq_len = T * (Hl // cfg.patch_size[1]) * (Wl // cfg.patch_size[2])

        cache = os.path.join(args.out, f"{shard}_clear_xT.pt")
        if os.path.exists(cache):
            x_T = torch.load(cache, map_location=device)
        else:
            x_T = invert(x0c)
            torch.save(x_T.cpu(), cache)
        x_base = regenerate(x_T)
        rec_err = ((x_base - x0c).norm() / x0c.norm()).item()
        print(f"{shard}: clear inverted + regenerated, rel-L2 {rec_err:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        dec_c = vae.decode([x0c])[0] if args.pixel or n_saved < args.save_videos else None
        dec_base = vae.decode([x_base])[0] if args.pixel or n_saved < args.save_videos else None

        for _, _, vid, params in pairs:
            x0w = lat[vid]
            pstr = ",".join(f"{k}{v:g}" for k, v in params.items() if v)
            if args.direction == 'global':
                dsel = [r for r in records if r['shard'] != shard]
            else:
                dsel = nearest_records(records, params, args.k, exclude_shards=[shard])
            dirs = build_directions(dsel)
            want_dec = args.pixel or n_saved < args.save_videos
            dec_w = vae.decode([x0w])[0] if want_dec else None

            arms = [('base', 0.0, None, x_base, None)]
            for w in windows:
                for a in alphas:
                    st = WeatherSteerer(dirs, alpha=a, blocks=blocks, sigma_window=w,
                                        norm_preserve=args.norm_preserve)
                    arms.append((f'a={a:g}|{w[0]:g}-{w[1]:g}', a, w, None, st))
            vids = []
            line = []
            for name, a, w, x_hat, st in arms:
                if x_hat is None:
                    x_hat = regenerate(x_T, st)
                m = latent_metrics(x_hat, x_base, x0c, x0w)
                row = dict(shard=shard, variant=vid, params=pstr, arm=name, alpha=a,
                           window=f"{w[0]:g}-{w[1]:g}" if w else '',
                           n_dir_pairs=dirs['n_pairs'],
                           n_applied=st.n_applied if st else 0, **m)
                if want_dec:
                    dec = dec_base if name == 'base' else vae.decode([x_hat])[0]
                    vids.append(dec)
                    if args.pixel:
                        row['psnr_weather'] = psnr(dec, dec_w)
                        row['psnr_clear'] = psnr(dec, dec_c)
                if st is not None and st.n_applied == 0:
                    print(f"WARNING {name}: steering never fired", flush=True)
                wr.writerow(row)
                fcsv.flush()
                for key in ('frac', 'cos', 'err', 'psnr_weather', 'psnr_clear'):
                    if key in row:
                        agg[(name, key)].append(row[key])
                line.append(f"{name}: frac {m['frac']:.2f} cos {m['cos']:.2f} err {m['err']:.2f}"
                            + (f" psnrW {row['psnr_weather']:.1f}" if 'psnr_weather' in row else ''))
            if n_saved < args.save_videos and vids:
                grid_v = torch.cat([panel(dec_c), panel(dec_w)] + [panel(v) for v in vids], dim=3)
                save_video(tensor=grid_v[None], save_file=os.path.join(
                    args.out, f"{shard}_{vid}.mp4"), fps=cfg.sample_fps, nrow=1,
                    normalize=True, value_range=(-1, 1))
                n_saved += 1
            print(f"  {shard}_{vid} ({pstr})\n    " + "\n    ".join(line), flush=True)
        del lat
        torch.cuda.empty_cache()
    fcsv.close()

    summary = {f"{arm}|{key}": dict(mean=sum(v) / len(v), n=len(v))
               for (arm, key), v in agg.items()}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print("\nmean over pairs:")
    for key in sorted(summary):
        print(f"  {key:32s} {summary[key]['mean']:.4f}  (n={summary[key]['n']})")
    print("PAIRED_EVAL_INV_DONE", flush=True)


if __name__ == '__main__':
    main()
