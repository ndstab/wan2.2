# Copyright 2024-2025. All rights reserved.
"""Quantify how DiT block representations change across blocks and timesteps.

Consumes the `captures.pt` dumps produced by `generate.py --capture_blocks ...`
and produces the (block x timestep) trend tables that eyeballing decoded PNGs
cannot give you.

Two families of metric, deliberately kept separate because they answer different
questions and have different confounds:

  1. CROSS-RUN divergence (needs --baseline and --injected).
     Per (block, t): how far the injected residual has moved from the baseline
     residual. Answers "where does injection change the representation".
     CONFOUND: from step ~2 onward the two runs' latents have already diverged,
     so a cell at (block 25, t 20) mixes this block's injection effect with
     accumulated trajectory drift. Read it as an upper bound, not an isolate.

  2. WITHIN-RUN block delta (works on a single run).
     Per (block_i, t): how much the residual changed between this captured block
     and the previously captured one. Answers "which depths do the most work,
     and when". Free of the drift confound because it never crosses runs.

Also reports per-cell token-norm statistics, which is how you spot a block whose
activations blow up or collapse under injection.

Usage:

    python analyze_captures.py \
        --baseline outputs/latent-inspection/baseline-<ts>/captures.pt \
        --injected outputs/latent-inspection/injected-<ts>/captures.pt \
        --output_dir outputs/latent-inspection/analysis

Single-run mode (within-run metrics only):

    python analyze_captures.py \
        --baseline outputs/latent-inspection/baseline-<ts>/captures.pt \
        --output_dir outputs/latent-inspection/analysis
"""
import argparse
import csv
import logging
import os

import torch


def _parse_args():
    p = argparse.ArgumentParser(
        description="Block x timestep trend analysis over captures.pt dumps.")
    p.add_argument("--baseline", type=str, required=True,
                   help="Path to the no-injection captures.pt.")
    p.add_argument("--injected", type=str, default=None,
                   help="Path to the injected captures.pt. If omitted, only "
                        "within-run metrics are computed.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory for CSVs and heatmaps.")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip matplotlib heatmaps, write CSVs only.")
    return p.parse_args()


def _load(path):
    """Load a captures dump, preferring mmap so an 8 GB dump doesn't land in RAM."""
    try:
        d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        logging.info(f"loaded (mmap) {path}")
        return d
    except Exception as exc:
        logging.warning(f"mmap load failed ({type(exc).__name__}), falling back "
                        f"to full load for {path}")
        return torch.load(path, map_location="cpu", weights_only=False)


def _valid_tokens(x, grid_sizes):
    """Strip the zero padding: only the first prod(grid) tokens carry signal.

    Captures are [seq_len, dim] where seq_len >= F_p*H_p*W_p. The tail is the
    zero pad added in WanModel.forward; including it would dilute every metric
    by a constant factor and make cells with different padding incomparable.
    """
    if grid_sizes is None:
        return x
    f, h, w = [int(v) for v in grid_sizes[0].tolist()]
    return x[:f * h * w]


def _pair_metrics(a, b):
    """a, b: [L, D] fp16 CPU. Returns dict of comparison metrics in fp32."""
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    diff = a - b

    rel_l2 = (diff.norm() / a.norm().clamp_min(1e-8)).item()

    # Mean per-token cosine similarity — scale-invariant, so it separates
    # "rotated the representation" from "rescaled it".
    num = (a * b).sum(dim=-1)
    den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)
    cos = (num / den)
    mean_cos = cos.mean().item()

    return {
        "rel_l2": rel_l2,
        "mean_cos": mean_cos,
        "cos_dist": 1.0 - mean_cos,
        "frac_tokens_cos_below_0p9": (cos < 0.9).float().mean().item(),
    }


def _norm_stats(x):
    x = x.to(torch.float32)
    n = x.norm(dim=-1)
    return {
        "token_norm_mean": n.mean().item(),
        "token_norm_std": n.std().item(),
        "act_absmax": x.abs().max().item(),
    }


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logging.info(f"wrote {path}")


def _heatmap(rows, value_key, blocks, timesteps, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(f"matplotlib unavailable ({type(exc).__name__}); "
                        f"skipping {path}")
        return

    lookup = {(r["block"], r["timestep"]): r[value_key] for r in rows}
    grid = [[lookup.get((b, t), float("nan")) for t in timesteps] for b in blocks]

    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(timesteps),
                                    1.0 + 0.5 * len(blocks)))
    im = ax.imshow(grid, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(timesteps)), [str(t) for t in timesteps])
    ax.set_yticks(range(len(blocks)), [str(b) for b in blocks])
    ax.set_xlabel("denoising step index (0 = noisiest)")
    ax.set_ylabel("DiT block index")
    ax.set_title(title)
    for i in range(len(blocks)):
        for j in range(len(timesteps)):
            v = grid[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f"wrote {path}")


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    base = _load(args.baseline)
    base_blocks = base["blocks"]
    grid = base.get("grid_sizes")
    keys = sorted(base_blocks.keys())
    block_ids = sorted({k[0] for k in keys})
    step_ids = sorted({k[1] for k in keys})
    logging.info(f"baseline: {len(keys)} cells, blocks={block_ids}, "
                 f"timesteps={step_ids}")

    # ---- within-run: how much does each captured depth change the residual ----
    within_rows = []
    for t in step_ids:
        prev = None
        prev_b = None
        for b in block_ids:
            if (b, t) not in base_blocks:
                continue
            cur = _valid_tokens(base_blocks[(b, t)], grid)
            row = {"block": b, "timestep": t, "prev_block": prev_b}
            row.update(_norm_stats(cur))
            if prev is not None:
                row.update(_pair_metrics(cur, prev))
            else:
                row.update({"rel_l2": float("nan"), "mean_cos": float("nan"),
                            "cos_dist": float("nan"),
                            "frac_tokens_cos_below_0p9": float("nan")})
            within_rows.append(row)
            prev, prev_b = cur, b

    _write_csv(
        os.path.join(args.output_dir, "within_run_block_delta.csv"),
        within_rows,
        ["block", "timestep", "prev_block", "rel_l2", "mean_cos", "cos_dist",
         "frac_tokens_cos_below_0p9", "token_norm_mean", "token_norm_std",
         "act_absmax"])

    if not args.no_plots:
        _heatmap(within_rows, "rel_l2", block_ids, step_ids,
                 "Within-run: relative L2 change vs previous captured block",
                 os.path.join(args.output_dir, "within_run_rel_l2.png"))
        _heatmap(within_rows, "token_norm_mean", block_ids, step_ids,
                 "Within-run: mean token norm",
                 os.path.join(args.output_dir, "within_run_token_norm.png"))

    # ---- cross-run: where does injection move the representation ----
    if args.injected is None:
        logging.info("no --injected given; skipping cross-run metrics")
        return

    inj = _load(args.injected)
    inj_blocks = inj["blocks"]
    common = sorted(set(base_blocks.keys()) & set(inj_blocks.keys()))
    missing = sorted(set(base_blocks.keys()) ^ set(inj_blocks.keys()))
    if missing:
        logging.warning(f"{len(missing)} cells present in only one run; "
                        f"ignoring them: {missing[:8]}...")
    logging.info(f"cross-run: {len(common)} comparable cells")

    cross_rows = []
    for (b, t) in common:
        a = _valid_tokens(base_blocks[(b, t)], grid)
        c = _valid_tokens(inj_blocks[(b, t)], inj.get("grid_sizes", grid))
        if a.shape != c.shape:
            logging.warning(f"shape mismatch at block={b} t={t}: "
                            f"{tuple(a.shape)} vs {tuple(c.shape)}; skipping")
            continue
        row = {"block": b, "timestep": t}
        row.update(_pair_metrics(c, a))  # injected vs baseline
        bs = _norm_stats(a)
        is_ = _norm_stats(c)
        row["baseline_token_norm_mean"] = bs["token_norm_mean"]
        row["injected_token_norm_mean"] = is_["token_norm_mean"]
        row["norm_ratio"] = (is_["token_norm_mean"]
                             / max(bs["token_norm_mean"], 1e-8))
        cross_rows.append(row)

    _write_csv(
        os.path.join(args.output_dir, "cross_run_divergence.csv"),
        cross_rows,
        ["block", "timestep", "rel_l2", "mean_cos", "cos_dist",
         "frac_tokens_cos_below_0p9", "baseline_token_norm_mean",
         "injected_token_norm_mean", "norm_ratio"])

    if not args.no_plots:
        cb = sorted({r["block"] for r in cross_rows})
        ct = sorted({r["timestep"] for r in cross_rows})
        _heatmap(cross_rows, "cos_dist", cb, ct,
                 "Cross-run: 1 - mean token cosine (injected vs baseline)",
                 os.path.join(args.output_dir, "cross_run_cos_dist.png"))
        _heatmap(cross_rows, "rel_l2", cb, ct,
                 "Cross-run: relative L2 (injected vs baseline)",
                 os.path.join(args.output_dir, "cross_run_rel_l2.png"))
        _heatmap(cross_rows, "norm_ratio", cb, ct,
                 "Cross-run: injected / baseline mean token norm",
                 os.path.join(args.output_dir, "cross_run_norm_ratio.png"))


if __name__ == "__main__":
    main()
