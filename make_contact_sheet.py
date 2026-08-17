# Copyright 2024-2025. All rights reserved.
"""Assemble decoded block/timestep frames into a single comparison figure.

Rows are DiT blocks, columns are denoising steps. Each cell stacks the baseline
frame above the injected frame so the pair is read vertically, which makes the
depth trend legible down a column and the timestep trend legible across a row.

    python make_contact_sheet.py \
        --decoded_dir outputs/latent-inspection/decoded \
        --baseline baseline-20260814_202759 \
        --injected injected-20260814_204155 \
        --out outputs/latent-inspection/analysis/contact_sheet.jpg
"""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont

CELL_W = 340
PAD = 6
LEFT_MARGIN = 58
TOP_MARGIN = 46
LABEL_H = 16


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _index(d):
    """Map (block, step) -> path for one arm's decoded frames."""
    out = {}
    for fn in os.listdir(d):
        m = re.match(r"block_(\d+)_t_(\d+)\.(jpg|png)$", fn)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = os.path.join(d, fn)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded_dir", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--injected", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = _index(os.path.join(args.decoded_dir, args.baseline))
    inj = _index(os.path.join(args.decoded_dir, args.injected))
    keys = sorted(set(base) & set(inj))
    blocks = sorted({k[0] for k in keys})
    steps = sorted({k[1] for k in keys})
    if not keys:
        raise SystemExit("no overlapping frames between the two arms")

    probe = Image.open(base[keys[0]])
    cell_h = round(CELL_W * probe.height / probe.width)
    pair_h = cell_h * 2 + LABEL_H

    W = LEFT_MARGIN + len(steps) * (CELL_W + PAD) + PAD
    H = TOP_MARGIN + len(blocks) * (pair_h + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f_hdr, f_lbl = _font(19), _font(13)

    d.text((PAD, 8), "each cell:  TOP = baseline (no injection)   |   "
                     "BOTTOM = injected (sandstorm, lambda=0.5, all 30 blocks)"
                     "      prompt 'car on city street', seed 42",
           fill="black", font=f_lbl)

    for j, t in enumerate(steps):
        x = LEFT_MARGIN + j * (CELL_W + PAD)
        d.text((x + CELL_W // 2 - 26, TOP_MARGIN - 20), f"step {t}",
               fill="black", font=f_hdr)

    for i, b in enumerate(blocks):
        y = TOP_MARGIN + i * (pair_h + PAD)
        d.text((6, y + pair_h // 2 - 10), f"blk\n{b:>3}", fill="black", font=f_hdr)
        for j, t in enumerate(steps):
            x = LEFT_MARGIN + j * (CELL_W + PAD)
            for row, src in enumerate((base, inj)):
                im = Image.open(src[(b, t)]).convert("RGB").resize(
                    (CELL_W, cell_h), Image.LANCZOS)
                sheet.paste(im, (x, y + row * cell_h))
            d.rectangle([x, y, x + CELL_W, y + cell_h * 2], outline="black")
            d.line([x, y + cell_h, x + CELL_W, y + cell_h], fill="red", width=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out, "JPEG", quality=88)
    print(f"wrote {args.out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
