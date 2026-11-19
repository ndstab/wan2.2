"""T5.1 / T5.3 -- paired-difference atlas on WeatherWeaver simulation pairs.

Each WeatherWeaver simulation shard is one scene along one camera path: a clear
render (all weather parameters 0) plus several renders of the *same* path under
known weather vectors. For a (clear, weather) pair we

  1. VAE-encode both videos exactly as WanTI2V.t2v() encodes a reference,
  2. noise both latents to sigma with the SAME epsilon,
  3. run one DiT forward pass on each with identical (null) conditioning,
  4. compare the post-block residuals of all 30 blocks.

Geometry, camera, frame index, noise, timestep and conditioning are identical
across the pair, so any representation difference is attributable to weather.
A single forward pass has no trajectory, so there is no accumulated drift --
every (block, sigma) cell is as clean as the step-0 column of the T1.1 atlas.

The same forward pass also gives the phase measurement (T5.3): the model's
x0-prediction for each member, and how much of the true weather difference it
preserves at each noise level.

No gradients, no optimiser, no weight changes.

Outputs (per pair) go to <out>/<shard>_<variant>.pt:
  per (block, sigma): rel diff, mean token cosine distance, token-mean delta
  vector [dim], per-token |delta| map [T, H/2, W/2]
  per sigma: x0-prediction difference ratios
"""
import argparse
import glob
import json
import os
import tarfile
import tempfile
import time

import torch

from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def load_video_latent(vae, path, frame_num, size, device):
    """Encode a clip the way t2v() encodes a reference, after a 16:9 centre crop.

    WeatherWeaver renders are 1920x1280 (3:2). The DiT runs at 1280x704, so we
    crop rows symmetrically to the target aspect *before* resizing, rather than
    squashing the frame -- both members of a pair get the identical crop.
    """
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    total = len(vr)
    idx = torch.linspace(0, total - 1, steps=frame_num).round().long()
    idx = idx.clamp(0, total - 1)
    frames = vr.get_batch(idx.tolist())                    # [F,H,W,C] uint8
    frames = frames.permute(3, 0, 1, 2).float() / 255.0    # [C,F,H,W]
    _, _, h, w = frames.shape
    tw, th = size
    keep_h = round(w * th / tw)
    if keep_h < h:
        top = (h - keep_h) // 2
        frames = frames[:, :, top:top + keep_h, :]
    return encode_frames(vae, frames, size, device)


def encode_frames(vae, frames, size, device):
    """[C,F,H,W] in [0,1] -> resize to size=(w,h) -> [-1,1] -> VAE latent."""
    tw, th = size
    frames = frames * 2.0 - 1.0
    frames = torch.nn.functional.interpolate(
        frames.permute(1, 0, 2, 3),
        size=(th, tw), mode='bilinear', align_corners=False,
    ).permute(1, 0, 2, 3).contiguous()
    frames = frames.to(device=device, dtype=torch.float32)
    return vae.encode([frames])[0].to(dtype=torch.float32)


# ---- WeatherWeaver `generation` split (SDXL + Prompt-to-Prompt image pairs) ----
# Each sample is a short mp4 whose frames are [clean | edited] side by side;
# frame 0's right half is (near-)unedited, frames 1..n-1 step through the
# strength levels listed in meta['params'] (frame i <-> params[i-1]). One
# weather factor per sample, unlike the simulation renders.

SKIP_PROMPT = ('360', '180-degree', 'panoram', 'fisheye', "bird's eye", 'aerial')


def image_samples(tar_path, tmpdir, max_samples=0, skip=SKIP_PROMPT):
    """Yield (sample_id, mp4_path, meta) from a generation-split tar, streaming.

    Samples whose clean prompt mentions a panoramic / aerial view are skipped:
    they are warped or third-person, the opposite of what the project needs.
    """
    pending, n = {}, 0
    with tarfile.open(tar_path, mode='r|') as t:     # sequential: no full scan
        for m in t:
            if not m.isfile():
                continue
            parts = [p for p in m.name.split('/') if p not in ('.', '')]
            if len(parts) < 2:
                continue
            sid, fname = parts[-2], parts[-1]
            if fname.startswith('._'):               # macOS AppleDouble sidecar
                continue
            slot = pending.setdefault(sid, {})
            if fname.endswith('.json'):
                slot['meta'] = json.load(t.extractfile(m))
            elif fname.endswith('.mp4'):
                dst = os.path.join(tmpdir, f"{sid}.mp4")
                with open(dst, 'wb') as f:
                    f.write(t.extractfile(m).read())
                slot['mp4'] = dst
            if 'meta' in slot and 'mp4' in slot:
                meta, mp4 = slot['meta'], slot['mp4']
                del pending[sid]
                if any(k in meta['prompts'][0].lower() for k in skip):
                    os.remove(mp4)
                    continue
                yield sid, mp4, meta
                n += 1
                if max_samples and n >= max_samples:
                    return


def image_levels(spec, n_frames):
    """'max' -> [last frame]; 'all' -> every edited frame; else frame indices."""
    if spec == 'max':
        return [n_frames - 1]
    if spec == 'all':
        return list(range(1, n_frames))
    idx = [int(i) for i in spec.split(',')]
    if any(i < 1 or i >= n_frames for i in idx):
        raise ValueError(f"image levels {idx} outside [1, {n_frames})")
    return idx


def load_image_pair_latents(vae, mp4, levels, size, device):
    """-> (clear latent, {frame_idx: edited latent}); each a 1-frame video."""
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(mp4, ctx=decord.cpu(0))
    frames = vr.get_batch(list(range(len(vr)))).float() / 255.0   # [F,H,W,C]
    half = frames.shape[2] // 2

    def enc(img):                                    # [H,W,C] -> latent
        return encode_frames(vae, img.permute(2, 0, 1)[:, None], size, device)

    return enc(frames[0, :, :half]), {i: enc(frames[i, :, half:]) for i in levels}


def shard_pairs(tar_path, tmpdir):
    """Yield (clear_mp4, variant_mp4, variant_id, params) for one shard."""
    with tarfile.open(tar_path) as t:
        names = t.getnames()
        meta = [n for n in names if n.endswith('.json')]
        params = json.load(t.extractfile(meta[0]))['params']
        paths = {}
        for n in names:
            if n.endswith('.mp4'):
                vid = os.path.basename(n)[:-4]
                dst = os.path.join(tmpdir, vid + '.mp4')
                with open(dst, 'wb') as f:
                    f.write(t.extractfile(n).read())
                paths[vid] = dst
    clear = [k for k, v in params.items() if all(x == 0 for x in v.values())]
    if len(clear) != 1:
        raise ValueError(f"{tar_path}: expected one clear video, found {clear}")
    clear = clear[0]
    for vid, p in sorted(params.items()):
        if vid != clear and vid in paths:
            yield paths[clear], paths[vid], vid, p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--shards', required=True, help='glob of WeatherWeaver .tar shards')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame_num', type=int, default=61)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--shift', type=float, default=5.0)
    ap.add_argument('--step_idx', default='0,5,10,15,20,25,30,35,40,45,49',
                    help='indices into the 50-step schedule to probe')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_shards', type=int, default=0)
    ap.add_argument('--pair_mode', choices=['weather', 'content'], default='weather',
                    help="weather: clear vs its weather variants (same scene). "
                         "content: clear of shard i vs clear of shard i+1 -- the "
                         "null control for cross-scene consistency")
    ap.add_argument('--dataset', choices=['sim', 'image'], default='sim',
                    help="sim: WeatherWeaver simulation_scene* video shards. "
                         "image: the generation split (SDXL image pairs, one "
                         "weather factor each), run as 1-frame videos")
    ap.add_argument('--image_size', type=int, default=960,
                    help='square side for --dataset image (multiple of 32; '
                         '960^2 ~ the 1280x704 training area)')
    ap.add_argument('--image_levels', default='max',
                    help="'max', 'all', or comma-separated frame indices")
    ap.add_argument('--max_samples', type=int, default=0,
                    help='--dataset image: samples per tar (0 = all)')
    args = ap.parse_args()
    if args.dataset == 'image' and args.pair_mode == 'content':
        raise ValueError("--pair_mode content is only defined for --dataset sim")

    cfg = ti2v_5B
    device = torch.device('cuda')
    param_dtype = cfg.param_dtype
    size = ((args.image_size, args.image_size) if args.dataset == 'image'
            else (1280, 704))
    os.makedirs(args.out, exist_ok=True)

    vae = Wan2_2_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
                     device=device)
    model = WanModel.from_pretrained(args.ckpt_dir)
    model = model.eval().requires_grad_(False).to(device).to(param_dtype)
    n_blocks = len(model.blocks)

    # Null conditioning for both members: the model sees only visual content,
    # so the analysis does not depend on a prompt.
    context = [torch.zeros(cfg.text_len, 4096, device=device, dtype=torch.float32)]

    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1,
        use_dynamic_shifting=False)
    sched.set_timesteps(args.sample_steps, device=device, shift=args.shift)
    sigmas_all = (sched.timesteps.float() / cfg.num_train_timesteps).tolist()
    step_idx = [int(s) for s in args.step_idx.split(',')]
    sigmas = [sigmas_all[i] for i in step_idx]

    # ---- block hooks: phase 'store' keeps clear outputs, 'compare' diffs ----
    state = {'phase': None, 'store': [None] * n_blocks, 'stats': [None] * n_blocks}

    def make_hook(b):
        def hook(_m, _inp, out):
            h = out[0] if out.dim() == 3 else out            # [L, D]
            if state['phase'] == 'store':
                state['store'][b] = h.detach().clone()
            elif state['phase'] == 'compare':
                hc = state['store'][b].float()
                hw = h.float()
                d = hw - hc
                state['stats'][b] = dict(
                    norm_c=hc.norm().item(),
                    norm_d=d.norm().item(),
                    cos_tok=(1 - torch.nn.functional.cosine_similarity(
                        hw, hc, dim=-1)).mean().item(),
                    dmean=d.mean(0).half().cpu(),
                    tokmap=d.norm(dim=-1).half().cpu(),
                )
        return hook

    handles = [blk.register_forward_hook(make_hook(b))
               for b, blk in enumerate(model.blocks)]

    def forward(x, sigma):
        t = torch.tensor([sigma * cfg.num_train_timesteps], device=device,
                         dtype=torch.float32)
        with torch.amp.autocast('cuda', dtype=param_dtype):
            v = model([x.to(param_dtype)], t=t, context=context,
                      seq_len=seq_len)[0]
        return v.float()

    shards = sorted(glob.glob(args.shards))
    if args.max_shards:
        shards = shards[:args.max_shards]
    print(f"{len(shards)} shards, sigmas {['%.3f' % s for s in sigmas]}", flush=True)

    tmp_root = os.environ.get('TMPDIR', '/tmp')

    def sim_jobs():
        """Yield (shard, lat, pairs, extra) for the simulation video shards."""
        if args.pair_mode == 'content':
            # (clear_i, clear_{i+1}) for i = 0, 2, 4, ...: two different scenes,
            # no weather on either. Non-overlapping, so no two control pairs
            # share a scene -- overlapping pairs would share a clip with
            # opposite sign and bias the cross-pair cosine negative.
            jobs = list(zip(shards[0::2], shards[1::2]))
        else:
            jobs = [(s, None) for s in shards]
        for tar_path, tar_other in jobs:
            shard = os.path.basename(tar_path)[:-4]
            with tempfile.TemporaryDirectory(dir=tmp_root) as td:
                pairs = list(shard_pairs(tar_path, td))
                lat = {}
                if tar_other is None:
                    for c_mp4, w_mp4, vid, _ in pairs:
                        if 'clear' not in lat:
                            lat['clear'] = load_video_latent(vae, c_mp4, args.frame_num, size, device)
                        lat[vid] = load_video_latent(vae, w_mp4, args.frame_num, size, device)
                else:
                    other = os.path.basename(tar_other)[:-4]
                    with tempfile.TemporaryDirectory(dir=td) as td2:
                        o_clear = next(shard_pairs(tar_other, td2))[0]
                        lat['clear'] = load_video_latent(vae, pairs[0][0], args.frame_num, size, device)
                        lat['vs' + other] = load_video_latent(vae, o_clear, args.frame_num, size, device)
                    pairs = [(None, None, 'vs' + other,
                              {k: 0.0 for k in pairs[0][3]})]
            yield shard, lat, pairs, {}

    def image_jobs():
        """Yield one job per generation-split sample: clean vs its edit(s).

        The 'shard' id is <effect>_<tar>_<sample>, so the cross-scene
        consistency analysis treats every sample as its own scene and never
        compares two strength levels of the same image against each other.
        """
        for tar_path in shards:
            effect = os.path.basename(os.path.dirname(os.path.abspath(tar_path)))
            tar_id = os.path.basename(tar_path)[:-4]
            with tempfile.TemporaryDirectory(dir=tmp_root) as td:
                for sid, mp4, meta in image_samples(tar_path, td, args.max_samples):
                    eff = meta.get('effect', effect)
                    import decord
                    n_frames = len(decord.VideoReader(mp4, ctx=decord.cpu(0)))
                    levels = image_levels(args.image_levels, n_frames)
                    clear, edits = load_image_pair_latents(vae, mp4, levels, size, device)
                    os.remove(mp4)
                    lat = {'clear': clear}
                    pairs = []
                    for i in levels:
                        vid = f"L{i}"
                        lat[vid] = edits[i]
                        pairs.append((None, None, vid, meta['params'][i - 1]))
                    yield (f"{eff}_{tar_id}_{sid}", lat, pairs,
                           dict(effect=eff, prompts=meta['prompts']))

    seq_len = None
    jobs_iter = image_jobs() if args.dataset == 'image' else sim_jobs()
    for si, (shard, lat, pairs, extra) in enumerate(jobs_iter):
        x0c = lat['clear']
        T, Hl, Wl = x0c.shape[1:]
        tok_shape = (T, Hl // cfg.patch_size[1], Wl // cfg.patch_size[2])
        seq_len = T * tok_shape[1] * tok_shape[2]

        for _, _, vid, params in pairs:
            out_path = os.path.join(args.out, f"{shard}_{vid}.pt")
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            x0w = lat[vid]
            true_diff = (x0w - x0c).norm().item()
            rec = dict(shard=shard, variant=vid, params=params, sigmas=sigmas,
                       step_idx=step_idx, tok_shape=tok_shape,
                       latent_diff=true_diff, blocks={}, x0pred={}, **extra)
            for s_i, sigma in enumerate(sigmas):
                g = torch.Generator(device=device).manual_seed(args.seed * 1000 + s_i)
                eps = torch.randn(x0c.shape, generator=g, device=device)
                xc = (1 - sigma) * x0c + sigma * eps
                xw = (1 - sigma) * x0w + sigma * eps

                state['phase'] = 'store'
                vc = forward(xc, sigma)
                state['phase'] = 'compare'
                vw = forward(xw, sigma)
                state['phase'] = None

                for b in range(n_blocks):
                    rec['blocks'][(b, s_i)] = state['stats'][b]
                # x0 = x_t - sigma * v. How much of the true weather difference
                # survives into the model's clean-image prediction at this sigma?
                x0c_hat = xc - sigma * vc
                x0w_hat = xw - sigma * vw
                pred_diff = x0w_hat - x0c_hat
                rec['x0pred'][s_i] = dict(
                    pred_over_true=pred_diff.norm().item() / true_diff,
                    err_over_true=(pred_diff - (x0w - x0c)).norm().item() / true_diff,
                    cos_pred_true=torch.nn.functional.cosine_similarity(
                        pred_diff.flatten(), (x0w - x0c).flatten(), dim=0).item(),
                )
            torch.save(rec, out_path)
            pstr = ",".join(f"{k}{v:g}" for k, v in params.items() if v)
            print(f"[{si + 1}] {shard}_{vid} ({pstr}) "
                  f"{time.time() - t0:.0f}s", flush=True)
        state['store'] = [None] * n_blocks
        del lat
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    print("PAIRED_ATLAS_DONE", flush=True)


if __name__ == '__main__':
    main()
