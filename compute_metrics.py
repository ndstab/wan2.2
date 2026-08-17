# Copyright 2024-2025. All rights reserved.
"""Style / content tradeoff metrics for injected videos, with no new dependencies.

Deliberately avoids CSD, LPIPS and scikit-image: `open_clip`, `lpips` and
`skimage` are all absent from the cluster env, and pip-installing them would
write into a filesystem sitting at its group quota. Everything here runs on
`torch` plus the OpenAI `clip` package with its already-cached ViT-L/14 weights,
so it works offline on a compute node.

Per video it reports:

  clip_style    cosine(CLIP(output), CLIP(reference))  - higher = more like the
                reference's appearance
  clip_content  cosine(CLIP(output), CLIP(no-injection baseline)) - higher =
                scene better preserved
  ssim          structural similarity vs the baseline, luminance, torch-native
  grad_ratio    mean gradient magnitude of output / of baseline. A proxy for how
                much fine detail survived; heavy haze drives this toward 0 even
                when the colour statistics look fine.

NOTE: clip_style is a CLIP cosine, i.e. the project's older WSIS-style measure,
NOT CSD. CSD needs `open_clip`; the CSD weights are already on disk at
`csd_weights/pytorch_model.bin`, so switching is a one-package job once the
quota issue is resolved.

Manifest format (CSV, no header):  label,video_path,baseline_path

    python compute_metrics.py --manifest m.csv --ref ref_videos/sandstorm.mp4 \
        --out metrics.csv
"""
import argparse
import csv
import os

# torch MUST be imported before decord: importing decord first segfaults on this
# cluster (observed 2026-08-17). wan/textimage2video.py sidesteps it the same way,
# by importing decord lazily inside the function that uses it.
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_frames(path, n=8):
    """Uniformly sample n frames -> [n, 3, H, W] float in [0, 1]."""
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    idx = torch.linspace(0, len(vr) - 1, n).round().long().clamp(0, len(vr) - 1)
    f = vr.get_batch(idx.tolist()).permute(0, 3, 1, 2).float() / 255.0
    return f


@torch.no_grad()
def clip_embed(model, frames, device):
    x = TF.resize(frames, [224, 224], antialias=True)
    x = ((x - CLIP_MEAN) / CLIP_STD).to(device)
    f = model.encode_image(x).float()
    f = F.normalize(f, dim=-1)
    return F.normalize(f.mean(0), dim=-1)   # average over frames, renormalise


def _gauss(ks=11, sigma=1.5):
    c = torch.arange(ks).float() - ks // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)


def ssim(a, b):
    """Mean SSIM over frames. a, b: [N, 3, H, W] in [0, 1]. Luminance only."""
    la = (0.299 * a[:, 0] + 0.587 * a[:, 1] + 0.114 * a[:, 2]).unsqueeze(1)
    lb = (0.299 * b[:, 0] + 0.587 * b[:, 1] + 0.114 * b[:, 2]).unsqueeze(1)
    w = _gauss().to(la.dtype)
    pad = w.shape[-1] // 2
    mu_a = F.conv2d(la, w, padding=pad)
    mu_b = F.conv2d(lb, w, padding=pad)
    saa = F.conv2d(la * la, w, padding=pad) - mu_a ** 2
    sbb = F.conv2d(lb * lb, w, padding=pad) - mu_b ** 2
    sab = F.conv2d(la * lb, w, padding=pad) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean().item()


def grad_energy(x):
    """Mean gradient magnitude - a scale-free proxy for surviving detail."""
    l = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    gx = l[:, :, :, 1:] - l[:, :, :, :-1]
    gy = l[:, :, 1:, :] - l[:, :, :-1, :]
    return (gx.abs().mean() + gy.abs().mean()).item() / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True,
                   help='CSV: label,video_path,baseline_path (no header)')
    p.add_argument('--ref', required=True, help='reference weather video')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    import clip
    model, _ = clip.load('ViT-L/14', device=device, jit=False)
    model.eval()

    ref_f = load_frames(args.ref, args.frames)
    ref_e = clip_embed(model, ref_f, device)

    rows = []
    with open(args.manifest) as fh:
        entries = [r for r in csv.reader(fh) if r and not r[0].startswith('#')]

    for label, vid, base in entries:
        if not os.path.exists(vid):
            print(f"[skip] missing {vid}")
            continue
        vf = load_frames(vid, args.frames)
        ve = clip_embed(model, vf, device)
        row = {'label': label,
               'clip_style': round(float(ve @ ref_e), 4)}
        if base and os.path.exists(base):
            bf = load_frames(base, args.frames)
            be = clip_embed(model, bf, device)
            if bf.shape == vf.shape:
                row['clip_content'] = round(float(ve @ be), 4)
                row['ssim'] = round(ssim(vf, bf), 4)
                ge_b = grad_energy(bf)
                row['grad_ratio'] = round(grad_energy(vf) / max(ge_b, 1e-8), 4)
            else:
                print(f"[warn] shape mismatch for {label}; skipping pixel metrics")
        rows.append(row)
        print(row)

    cols = ['label', 'clip_style', 'clip_content', 'ssim', 'grad_ratio']
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
