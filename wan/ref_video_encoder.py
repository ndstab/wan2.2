import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class RefVideoEncoder:

    def __init__(self, vae, target_dim: int):
        self.vae = vae

        vae_latent_channels = getattr(getattr(vae, "model", None), "z_dim", None)
        if vae_latent_channels is None:
            raise ValueError(
                "Unable to infer VAE latent channels (expected `vae.model.z_dim`)."
            )

        self.proj = nn.Linear(vae_latent_channels, target_dim)
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
        device = getattr(
            self.vae, "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Load and sample frames → [8, C, H, W] in [0, 1]
        frames = self._load_and_sample_frames(video_path, num_frames=33)

        # 2. Resize to 64×64
        frames = TF.resize(frames, [64, 64])             # [8, C, 64, 64]

        # 3. Normalize to [-1, 1]  — only once, frames are already in [0, 1]
        frames = frames * 2.0 - 1.0                      # [8, C, 64, 64]
        frames = frames.to(device=device, dtype=torch.float32)

        # 4. Pass ALL frames as a single clip through the 3D VAE
        #    Wan VAE expects input as a list of videos, each [C, T, H, W]
        video_clip = frames.permute(1, 0, 2, 3)          # [C, T=8, H, W]
        latents = self.vae.encode([video_clip])           # list of 1 latent
        latent = latents[0]                               # [C_latent, T_latent, H_latent, W_latent]
        #    With temporal stride 4: T_latent = ceil((8-1)/4)+1 = 3 (Wan formula)
        #    Spatial stride 8:       H_latent = 64/8 = 8, W_latent = 8

        # 5. Spatial average pool to 8×8 (in case spatial dim differs)
        C_lat, T_lat, H_lat, W_lat = latent.shape
        latent_2d = latent.reshape(C_lat * T_lat, H_lat, W_lat).unsqueeze(0)
        latent_2d = F.adaptive_avg_pool2d(latent_2d, (8, 8))  # [1, C*T, 8, 8]
        latent_2d = latent_2d.squeeze(0)                       # [C*T, 8, 8]

        # 6. Flatten to token sequence
        #    tokens = T_lat * 8 * 8  (dynamic, not hardcoded to 512)
        latent_2d = latent_2d.permute(1, 2, 0)                # [8, 8, C*T]
        num_tokens = 8 * 8
        c_combined = C_lat * T_lat
        tokens_flat = latent_2d.reshape(1, num_tokens, c_combined)  # [1, 64, C*T]

        # 7. Project to model context dim
        #    projection input dim may now be C_lat * T_lat instead of C_lat
        #    handle this by re-creating proj if needed
        if tokens_flat.shape[-1] != self.proj.in_features:
            self.proj = nn.Linear(tokens_flat.shape[-1], self.proj.out_features)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.proj.eval()

        self.proj = self.proj.to(device=device, dtype=tokens_flat.dtype)
        tokens = self.proj(tokens_flat)                        # [1, 64, target_dim]

        # 8. Sequence length
        ref_lens = torch.tensor(
            [tokens.size(1)], dtype=torch.long, device=tokens.device
        )

        return tokens, ref_lens