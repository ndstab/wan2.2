"""T5.5 (K/V) -- paired ground-truth evaluation of reference K/V injection,
optionally gated to a sigma window.

Same protocol as paired_eval_inv.py (exact fixed-point inversion of the clear
video, then regenerate), but the intervention is the project's own method:
self-attention K/V blending with a reference weather video (model.py),
re-noised to the current sigma with fresh noise every step, exactly as in
WanTI2V.t2v().

Reference choice -- no leakage: for a held-out pair (clear S, weather V with
weather vector p), the reference is the weather render from a DIFFERENT shard
whose weather vector is nearest to p. Its scene and camera path differ, so it
can only transfer weather. A `leak` arm (the target's own weather video as the
reference) is reported separately as a ceiling, never as a result.

Arms per pair:
  base                    regenerate, no injection (~ the clear video)
  kv=<lam>|<lo>-<hi>      inject with lambda lam, only while lo <= sigma <= hi
  leak=<lam>|<lo>-<hi>    same, reference = the target itself (ceiling)

Latent metrics (frac, cos, err) and PSNR as in paired_eval.py.
No gradients, no optimiser, no weight changes.
"""
import argparse
import csv
import glob
import json
import os
import tarfile
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
from wan.utils.steering import PARAM_KEYS


def shard_params(tar_path):
    """{variant_id: params} for one simulation shard, reading only its json."""
    with tarfile.open(tar_path) as t:
        meta = next(n for n in t.getnames() if n.endswith('.json'))
        m = json.load(t.extractfile(meta))
    return m['params'], m.get('params_value_max', {})


def extract_video(tar_path, vid, dst_dir):
    with tarfile.open(tar_path) as t:
        name = next(n for n in t.getnames() if os.path.basename(n) == f"{vid}.mp4")
        dst = os.path.join(dst_dir, f"{os.path.basename(tar_path)[:-4]}_{vid}.mp4")
        with open(dst, 'wb') as f:
            f.write(t.extractfile(name).read())
    return dst


def nearest_reference(pool, target, exclude_shard, scale):
    """pool: [(shard, vid, params, tar)]; nearest weather vector, other shard."""
    def vec(p):
        return torch.tensor([float(p.get(k, 0.0)) / scale.get(k, 1.0) for k in PARAM_KEYS])
    tv = vec(target)
    cands = [c for c in pool if c[0] != exclude_shard]
    return min(cands, key=lambda c: (vec(c[2]) - tv).norm().item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--shards', required=True, help='glob of eval shards')
    ap.add_argument('--ref_pool', required=True, help='glob of shards references may come from')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lambdas', default='0.05,0.1,0.2')
    ap.add_argument('--windows', default='0,1;0.9,1;0,0.9',
                    help="';'-separated 'lo,hi' sigma windows")
    ap.add_argument('--leak', default='0.1|0,1', help="'lam|lo,hi' for the ceiling arm, '' to skip")
    ap.add_argument('--inject_blocks', default='', help='comma list; empty = all 30')
    ap.add_argument('--fp_iters', type=int, default=3)
    ap.add_argument('--xT_cache', default='', help='dir holding <shard>_clear_xT.pt to reuse')
    ap.add_argument('--pixel', action='store_true')
    ap.add_argument('--save_videos', type=int, default=4)
    ap.add_argument('--ref_seed', type=int, default=7)
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
    inject_blocks = [int(b) for b in args.inject_blocks.split(',')] if args.inject_blocks else None

    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1,
        use_dynamic_shifting=False)
    sched.set_timesteps(args.sample_steps, device=device, shift=args.shift)
    grid = [0.0] + (sched.timesteps.float() / cfg.num_train_timesteps).tolist()[::-1]
    n_steps = len(grid) - 1
    seq_len = None

    def velocity(x, sigma, ref=None, lam=0.0):
        s = max(sigma, grid[1])
        t = torch.tensor([s * cfg.num_train_timesteps], device=device)
        kw = {}
        if ref is not None and lam > 0:
            kw = dict(ref_latents=[ref.to(cfg.param_dtype)], lambda_ref=float(lam),
                      inject_blocks=inject_blocks)
        with torch.amp.autocast('cuda', dtype=cfg.param_dtype):
            v = model([x.to(cfg.param_dtype)], t=t, context=context,
                      seq_len=seq_len, **kw)[0]
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

    def regenerate(x_T, ref0=None, lam=0.0, window=(0.0, 1.0)):
        g = torch.Generator(device=device).manual_seed(args.ref_seed)
        x, n_inj = x_T, 0
        for i in range(n_steps, 0, -1):
            s = grid[i]
            ref = None
            if ref0 is not None:
                # fresh noise every step, drawn even outside the window so the
                # noise stream does not depend on the window (as in t2v)
                eps = torch.randn(ref0.shape, generator=g, device=device)
                if window[0] <= s <= window[1]:
                    ref = (1 - s) * ref0 + s * eps
                    n_inj += 1
            x = x + (grid[i - 1] - s) * velocity(x, s, ref, lam)
        return x, n_inj

    lambdas = [float(v) for v in args.lambdas.split(',')]
    windows = [tuple(float(v) for v in w.split(',')) for w in args.windows.split(';')]
    leak = None
    if args.leak:
        lam_s, w_s = args.leak.split('|')
        leak = (float(lam_s), tuple(float(v) for v in w_s.split(',')))

    pool, scale = [], {}
    for tar in sorted(glob.glob(args.ref_pool)):
        params, pmax = shard_params(tar)
        scale.update(pmax)
        for vid, p in params.items():
            if any(float(v) for v in p.values()):
                pool.append((os.path.basename(tar)[:-4], vid, p, tar))
    print(f"reference pool: {len(pool)} weather clips", flush=True)

    fields = ['shard', 'variant', 'params', 'ref', 'ref_params', 'arm', 'lam', 'window',
              'n_inj', 'frac', 'cos', 'err', 'psnr_weather', 'psnr_clear']
    # One CSV per job: concurrent jobs appending to a single file interleave
    # their header lines with data rows (2026-10-20).
    tag = os.path.basename(args.shards).replace('*', 'all').replace('.tar', '')
    csv_path = os.path.join(args.out, f'metrics_{tag}.csv')
    # Resume: a pair counts as done once all its arms are logged, in this job's
    # CSV or in any earlier CSV in --out (parsed by position, header lines skipped).
    n_arms = 1 + len(lambdas) * len(windows) + (1 if leak else 0)
    cnt = defaultdict(int)
    for old in glob.glob(os.path.join(args.out, 'metrics*.csv')):
        for row in csv.reader(open(old)):
            if len(row) >= 2 and row[0] != 'shard':
                cnt[(row[0], row[1])] += 1
    done = {k for k, c in cnt.items() if c >= n_arms}
    new = not os.path.exists(csv_path)
    fcsv = open(csv_path, 'a', newline='')
    wr = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        wr.writeheader()
    agg = defaultdict(list)
    ref_cache = {}
    n_saved = 0
    pstr = lambda p: ",".join(f"{k}{float(v):g}" for k, v in p.items() if float(v))

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

        cache = os.path.join(args.xT_cache or args.out, f"{shard}_clear_xT.pt")
        if os.path.exists(cache):
            x_T = torch.load(cache, map_location=device)
        else:
            x_T = invert(x0c)
            torch.save(x_T.cpu(), os.path.join(args.out, f"{shard}_clear_xT.pt"))
        x_base, _ = regenerate(x_T)
        print(f"{shard}: base regenerated, rel-L2 "
              f"{((x_base - x0c).norm() / x0c.norm()).item():.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        want_dec = args.pixel or n_saved < args.save_videos
        # Decoded videos are ~0.7 GB each at 61 x 1280 x 704; keep them on the
        # CPU or a pair's worth of arms exhausts a 46 GB card (2026-10-20 OOM).
        dec_c = vae.decode([x0c])[0].cpu() if want_dec else None
        dec_base = vae.decode([x_base])[0].cpu() if want_dec else None

        for _, _, vid, params in pairs:
            if (shard, vid) in done:
                print(f"  {shard}_{vid}: already in metrics.csv, skipping", flush=True)
                continue
            x0w = lat[vid]
            rs, rv, rp, rtar = nearest_reference(pool, params, shard, scale)
            key = (rs, rv)
            if key not in ref_cache:
                with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', '/tmp')) as td:
                    ref_cache[key] = load_video_latent(
                        vae, extract_video(rtar, rv, td), args.frame_num, size, device)
            ref0 = ref_cache[key]
            dec_w = vae.decode([x0w])[0].cpu() if want_dec else None

            arms = [('base', 0.0, None, None)]
            for w in windows:
                for lam in lambdas:
                    arms.append((f'kv={lam:g}|{w[0]:g}-{w[1]:g}', lam, w, ref0))
            if leak:
                arms.append((f'leak={leak[0]:g}|{leak[1][0]:g}-{leak[1][1]:g}',
                             leak[0], leak[1], x0w))
            vids, lines = [], []
            for name, lam, w, ref in arms:
                if ref is None:
                    x_hat, n_inj = x_base, 0
                else:
                    x_hat, n_inj = regenerate(x_T, ref, lam, w)
                m = latent_metrics(x_hat, x_base, x0c, x0w)
                row = dict(shard=shard, variant=vid, params=pstr(params),
                           ref=f"{rs}_{rv}", ref_params=pstr(rp), arm=name, lam=lam,
                           window=f"{w[0]:g}-{w[1]:g}" if w else '', n_inj=n_inj, **m)
                if want_dec:
                    dec = dec_base if name == 'base' else vae.decode([x_hat])[0].cpu()
                    vids.append(dec)
                    if args.pixel:
                        row['psnr_weather'] = psnr(dec, dec_w)
                        row['psnr_clear'] = psnr(dec, dec_c)
                if ref is not None and n_inj == 0:
                    print(f"WARNING {name}: window never matched the schedule", flush=True)
                wr.writerow(row)
                fcsv.flush()
                for k in ('frac', 'cos', 'err', 'psnr_weather', 'psnr_clear'):
                    if k in row:
                        agg[(name, k)].append(row[k])
                lines.append(f"{name}: frac {m['frac']:.2f} cos {m['cos']:.2f} err {m['err']:.2f}"
                             + (f" psnrW {row['psnr_weather']:.1f} psnrC {row['psnr_clear']:.1f}"
                                if 'psnr_weather' in row else '') + f" n_inj {n_inj}")
            if n_saved < args.save_videos and vids:
                dec_ref = vae.decode([ref0])[0].cpu()
                g = torch.cat([panel(dec_c), panel(dec_w), panel(dec_ref)] +
                              [panel(v) for v in vids], dim=3)
                save_video(tensor=g[None], save_file=os.path.join(args.out, f"{shard}_{vid}.mp4"),
                           fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                n_saved += 1
            print(f"  {shard}_{vid} ({pstr(params)})  ref {rs}_{rv} ({pstr(rp)})\n    "
                  + "\n    ".join(lines), flush=True)
            del vids
            torch.cuda.empty_cache()
        del lat
        torch.cuda.empty_cache()
    fcsv.close()

    summary = {f"{a}|{k}": dict(mean=sum(v) / len(v), n=len(v)) for (a, k), v in agg.items()}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print("PAIRED_EVAL_KV_DONE", flush=True)


if __name__ == '__main__':
    main()
