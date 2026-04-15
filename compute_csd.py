"""Compute CSD (Contrastive Style Descriptors) similarity and optional SSIM.

CSD ViT-L weights: https://github.com/learn2phoenix/CSD — the authors host the
checkpoint on Google Drive. Download it manually on a node with internet access
and pass the local path via --csd_ckpt. This script loads a ViT-L/14 OpenCLIP
visual backbone and swaps in the CSD state dict.
"""
import argparse

import decord
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def load_video_frames(path, n=8):
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    idx = torch.linspace(0, len(vr) - 1, n).round().long().clamp(0, len(vr) - 1)
    frames = vr.get_batch(idx.tolist())  # [N, H, W, C] uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]
    frames = TF.resize(frames, [224, 224])
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (frames - mean) / std


def load_csd_model(ckpt_path, device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
    visual = model.visual
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    # CSD ships its backbone under a `backbone.` prefix; strip if present.
    sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    if missing:
        print(f"[CSD] missing keys: {len(missing)} (first few: {missing[:3]})")
    if unexpected:
        print(f"[CSD] unexpected keys: {len(unexpected)} (first few: {unexpected[:3]})")
    visual = visual.to(device).eval()
    return visual


@torch.no_grad()
def csd_embed(visual, frames, device):
    frames = frames.to(device)
    feats = visual(frames)
    feats = F.normalize(feats, dim=-1)
    return feats.mean(dim=0)  # avg over frames


@torch.no_grad()
def ssim_between_videos(path_a, path_b, n=8):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("scikit-image not installed — skipping SSIM.")
        return None
    import numpy as np
    fa = load_video_frames_raw(path_a, n)
    fb = load_video_frames_raw(path_b, n)
    scores = []
    for a, b in zip(fa, fb):
        scores.append(ssim(a, b, channel_axis=2, data_range=1.0))
    return float(np.mean(scores))


def load_video_frames_raw(path, n=8, size=(256, 256)):
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    idx = torch.linspace(0, len(vr) - 1, n).round().long().clamp(0, len(vr) - 1)
    frames = vr.get_batch(idx.tolist()).float() / 255.0  # [N, H, W, C]
    frames = frames.permute(0, 3, 1, 2)
    frames = TF.resize(frames, list(size))
    return frames.permute(0, 2, 3, 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_video', required=True)
    p.add_argument('--ref_video', required=True)
    p.add_argument('--baseline_video', default=None,
                   help="Optional baseline (no-ref) generation for content SSIM.")
    p.add_argument('--csd_ckpt', required=True, help="Path to CSD ViT-L weights.")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visual = load_csd_model(args.csd_ckpt, device)

    out_frames = load_video_frames(args.output_video)
    ref_frames = load_video_frames(args.ref_video)

    out_emb = csd_embed(visual, out_frames, device)
    ref_emb = csd_embed(visual, ref_frames, device)

    score = F.cosine_similarity(out_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
    print(f"CSD score: {score:.4f}")

    if args.baseline_video is not None:
        s = ssim_between_videos(args.output_video, args.baseline_video)
        if s is not None:
            print(f"Content SSIM: {s:.4f}")


if __name__ == '__main__':
    main()
