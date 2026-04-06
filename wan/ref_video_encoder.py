import math
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF


class RefVideoEncoder:

    def __init__(self, vae, target_dim: int):
        self.vae = vae

        self.proj = nn.Linear(768, target_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.proj.eval()

    def _load_and_sample_frames(self, video_path: str, num_frames: int = 8):
        import decord
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        if total_frames >= num_frames:
            indices = torch.linspace(0, total_frames - 1, steps=num_frames)
            indices = indices.round().long().clamp(0, total_frames - 1)
        else:
            indices = torch.arange(num_frames) % total_frames
        frames = vr.get_batch(indices.tolist())   # [N, H, W, C], uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W], [0, 1]
        return frames

    @torch.no_grad()
    def encode(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        import clip

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        frames = self._load_and_sample_frames(video_path, num_frames=8)
        frames = TF.resize(frames, [224, 224])

        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        frames = (frames - clip_mean) / clip_std
        frames = frames.to(device=device, dtype=torch.float32)

        # clip_model, _ = clip.load("ViT-L/14", device=device)
        import os
        clip_model, _ = clip.load(  
        "ViT-L/14", 
        device=device,
        download_root=os.path.expanduser("~/.cache/clip")
        )
        clip_model.eval()

        frame_features = clip_model.encode_image(frames)
        frame_features = frame_features.view(1, 8, 768)

        self.proj = self.proj.to(device=device, dtype=frame_features.dtype)
        self.proj.eval()
        tokens = self.proj(frame_features)

        ref_lens = torch.tensor([8], dtype=torch.long, device=tokens.device)

        return tokens, ref_lens