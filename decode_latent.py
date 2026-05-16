# Copyright 2024-2025. All rights reserved.
"""Decode captured intermediate-block latents into viewable PNG frames.

For each `(block_idx, timestep_idx)` tensor in a `captures.pt` dump produced by
`generate.py --capture_blocks ... --capture_timesteps ... --capture_dir ...`,
this script:

  1. Re-derives the timestep modulation `e` from the saved scalar `t`.
  2. Applies `model.head(x, e)` to convert token-space → patch-space.
  3. Applies `model.unpatchify(...)` to reach VAE latent space.
  4. Decodes through the VAE to pixel space.
  5. Saves the middle frame as a PNG.

The captured residual is a *mid-block* signal; decoding it through `head` treats
it as if it were the final residual. The resulting PNG is a visualization of
"what does the model 'think' the frame looks like at depth N, step t" — useful
for diagnostic inspection, not a faithful intermediate reconstruction.

Usage (single GPU; runs offline, after generation):

    python decode_latent.py \
        --captures outputs/latent-inspection/baseline-<ts>/captures.pt \
        --ckpt_dir /path/to/Wan2.2-TI2V-5B \
        --output_dir outputs/latent-inspection/baseline-<ts>/decoded \
        --task ti2v-5B
"""
import argparse
import logging
import math
import os

import torch
from PIL import Image

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel, sinusoidal_embedding_1d
from wan.modules.vae2_2 import Wan2_2_VAE


def _parse_args():
    p = argparse.ArgumentParser(
        description="Decode captured DiT block residuals to PNG frames.")
    p.add_argument("--captures", type=str, required=True,
                   help="Path to captures.pt produced by generate.py.")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Path to the Wan2.2-TI2V-5B checkpoint directory "
                        "(same one passed to generate.py).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write decoded PNGs into.")
    p.add_argument("--task", type=str, default="ti2v-5B",
                   choices=list(WAN_CONFIGS.keys()),
                   help="Task config to load (default: ti2v-5B).")
    p.add_argument("--frame_index", type=int, default=-1,
                   help="Which pixel-space frame index to save per capture. "
                        "Default -1 = middle frame.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu (cpu will be very slow).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only decode the first N captures (for quick tests).")
    return p.parse_args()


def _ensure_dim_match(captures, model):
    meta = captures['meta']
    if meta['dim'] != model.dim:
        raise ValueError(
            f"Dim mismatch: captures.dim={meta['dim']} vs model.dim={model.dim}. "
            "Is --ckpt_dir / --task the same one used to produce the captures?")
    if int(meta['num_layers']) != int(model.num_layers):
        raise ValueError(
            f"num_layers mismatch: captures={meta['num_layers']} vs "
            f"model={model.num_layers}.")
    if tuple(meta['patch_size']) != tuple(model.patch_size):
        raise ValueError(
            f"patch_size mismatch: captures={meta['patch_size']} vs "
            f"model={model.patch_size}.")


@torch.no_grad()
def _make_e_for_step(model, t_value, seq_len, device):
    """Re-derive the pre-projection modulation `e` from a scalar timestep `t`.

    Matches the construction inside WanModel.forward exactly so that head() sees
    the same modulation it would see during generation at this step.
    """
    t = torch.full((1, seq_len), float(t_value), dtype=torch.float32, device=device)
    bt = t.size(0)
    flat = t.flatten()
    e = model.time_embedding(
        sinusoidal_embedding_1d(model.freq_dim, flat).unflatten(0, (bt, seq_len)).float()
    )
    return e


@torch.no_grad()
def _decode_one(model, vae, x_tokens, t_value, grid_sizes, seq_len, device, dtype):
    """x_tokens: [seq_len, dim] (fp16, cpu). Returns pixel-space tensor [3, F, H, W]."""
    x = x_tokens.unsqueeze(0).to(device=device, dtype=dtype)  # [1, L, dim]

    e = _make_e_for_step(model, t_value, seq_len, device)  # [1, L, dim] fp32
    # head signature: (x, e) -> [1, L, out_dim * prod(patch_size)]
    x = model.head(x, e)
    # unpatchify expects a list-iterable along batch
    out_list = model.unpatchify(x, grid_sizes.to(device))  # list of [C, F, H, W]
    latent = out_list[0].to(dtype=torch.float32).unsqueeze(0)  # [1, C, F, H, W]? No — single sample [C, F, H, W]
    # VAE wants list of [C, F, H, W]
    pixels = vae.decode([out_list[0].to(dtype=torch.float32)])
    return pixels[0]  # [3, F, H, W] in [-1, 1] roughly


def _tensor_frame_to_png(frame, path):
    """frame: [3, H, W] in approx [-1, 1]. Clamp & save."""
    f = frame.detach().to(dtype=torch.float32).cpu()
    f = (f.clamp(-1.0, 1.0) + 1.0) * 0.5  # → [0, 1]
    f = (f * 255.0).round().clamp(0, 255).to(torch.uint8)
    arr = f.permute(1, 2, 0).numpy()  # [H, W, 3]
    Image.fromarray(arr).save(path)


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.exists(args.captures):
        raise FileNotFoundError(args.captures)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu":
        logging.warning("Running on CPU — VAE decode will be very slow.")

    cfg = WAN_CONFIGS[args.task]
    dtype = cfg.param_dtype

    logging.info(f"Loading captures: {args.captures}")
    captures = torch.load(args.captures, map_location="cpu", weights_only=False)
    meta = captures['meta']
    grid_sizes = captures['grid_sizes']
    seq_len = int(captures['seq_len'])
    t_values = captures['timestep_values']
    blocks = captures['blocks']

    logging.info(
        f"Captures: {len(blocks)} tensors across "
        f"blocks={meta['capture_blocks']}, "
        f"timesteps={meta['capture_timesteps']}, "
        f"seq_len={seq_len}, dim={meta['dim']}")

    # Load WanModel (we only need head/time_embedding/unpatchify, but loading the
    # full thing is the simplest path and matches the production code).
    logging.info(f"Loading WanModel from {args.ckpt_dir}")
    model = WanModel.from_pretrained(args.ckpt_dir)
    model.eval().requires_grad_(False)
    _ensure_dim_match(captures, model)
    model = model.to(device)

    logging.info(f"Loading VAE from {args.ckpt_dir}/{cfg.vae_checkpoint}")
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        device=device)

    # Deterministic order: sort by (block, timestep).
    keys = sorted(blocks.keys())
    if args.limit is not None:
        keys = keys[:args.limit]

    with torch.amp.autocast(device.type, dtype=dtype, enabled=(device.type == 'cuda')):
        for (block_idx, t_idx) in keys:
            if t_idx not in t_values:
                logging.warning(
                    f"Missing timestep_value for t_idx={t_idx}; skipping "
                    f"block={block_idx}")
                continue
            t_value = t_values[t_idx]
            x_tokens = blocks[(block_idx, t_idx)]
            try:
                pixels = _decode_one(
                    model, vae, x_tokens, t_value, grid_sizes, seq_len,
                    device, dtype)
            except Exception as exc:
                logging.error(
                    f"decode failed for block={block_idx} t_idx={t_idx}: {exc}")
                continue

            F_pix = pixels.shape[1]
            f_idx = (F_pix // 2) if args.frame_index < 0 else min(args.frame_index, F_pix - 1)
            frame = pixels[:, f_idx]
            out_path = os.path.join(
                args.output_dir,
                f"block_{block_idx:02d}_t_{t_idx:02d}.png")
            _tensor_frame_to_png(frame, out_path)
            logging.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
