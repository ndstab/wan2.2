import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.io import read_video
from torchvision.transforms import functional as TF


class RefVideoEncoder:

    def __init__(self, vae, target_dim: int):
        r"""
        Utility to encode a reference video into a compact token sequence that
        can be injected into Wan cross-attention as additional context.

        Args:
            vae: Loaded Wan VAE wrapper (e.g., Wan2_1_VAE instance).
            target_dim: Target embedding dimension to match the model's context dim.
        """
        self.vae = vae

        # infer latent channel dimension from underlying VAE model
        vae_latent_channels = getattr(getattr(vae, "model", None), "z_dim", None)
        if vae_latent_channels is None:
            raise ValueError("Unable to infer VAE latent channels (expected `vae.model.z_dim`).")

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
    frames = vr.get_batch(indices.tolist())  # [N, H, W, C]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]
    return frames

    @torch.no_grad()
    def encode(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode a reference video into a compact sequence of tokens suitable for
        cross-attention context injection.

        Steps:
            1. Load the video and uniformly sample 8 frames.
            2. Resize each frame to 64x64.
            3. Normalize pixel values to [-1, 1].
            4. Encode frames through the frozen VAE encoder.
            5. Apply spatial average pooling to 8x8.
            6. Flatten across frames and spatial dimensions to obtain 512 tokens.
            7. Project tokens to the model's context dimension.

        Returns:
            ref_tokens: Tensor of shape [1, 512, target_dim]
            ref_lens:   LongTensor of shape [1] with value 512
        """
        device = getattr(self.vae, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Load and sample frames
        frames = self._load_and_sample_frames(video_path, num_frames=8)  # [8, 3, H, W]

        # 2. Resize to 64x64
        frames = TF.resize(frames, [64, 64])  # [8, 3, 64, 64]

        # 3. Normalize to [-1, 1]
        frames = frames.to(device=device, dtype=torch.float32) / 255.0
        frames = frames * 2.0 - 1.0

        # 4. Encode through frozen VAE encoder
        # Treat each frame as a separate 1-frame video: [C, T, H, W] with T=1
        videos = [frame.unsqueeze(1) for frame in frames]  # list of [3, 1, 64, 64]
        latents_list = self.vae.encode(videos)

        # Each element in latents_list has shape [C_latent, T_latent, H_latent, W_latent]
        # We collapse the temporal dimension by averaging, yielding [C_latent, H_latent, W_latent]
        processed = []
        for latent in latents_list:
            if latent.dim() != 4:
                raise ValueError(
                    f"Expected VAE latents with 4 dimensions [C_latent, T, H, W], got shape {tuple(latent.shape)}"
                )
            if latent.size(1) > 1:
                latent = latent.mean(dim=1)
            else:
                latent = latent.squeeze(1)
            processed.append(latent)

        latents = torch.stack(processed, dim=0)  # [8, C_latent, H_latent, W_latent]

        # 5. Spatial average pooling to 8x8
        latents = F.adaptive_avg_pool2d(latents, (8, 8))  # [8, C_latent, 8, 8]

        # 6. Reshape to token sequence
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [8, 8, 8, C_latent]
        b, f, h, c_latent = latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3]
        assert b * f * h == 8 * 8 * 8, "Unexpected latent shape while creating reference tokens."
        latents = latents.view(1, 8 * 8 * 8, c_latent)  # [1, 512, C_latent]

        # 7. Project to model context dim
        self.proj.to(device=latents.device)
        tokens = self.proj(latents)  # [1, 512, target_dim]

        # 8. Sequence lengths tensor
        ref_lens = torch.tensor([tokens.size(1)], dtype=torch.long, device=tokens.device)

        return tokens, ref_lens

