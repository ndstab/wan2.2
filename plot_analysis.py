# Copyright 2024-2025. All rights reserved.
"""Render the CSVs from analyze_captures.py into (block x timestep) heatmaps.

Kept separate from analyze_captures.py because the cluster `torch` env has no
matplotlib; the intended flow is: compute CSVs on the cluster, copy the (tiny)
CSVs anywhere with matplotlib, plot here.

    python plot_analysis.py --analysis_dir outputs/latent-inspection/analysis
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _grid(rows, key):
    blocks = sorted({int(r["block"]) for r in rows})
    steps = sorted({int(r["timestep"]) for r in rows})
    d = {}
    for r in rows:
        try:
            d[(int(r["block"]), int(r["timestep"]))] = float(r[key])
        except (ValueError, KeyError):
            pass
    g = [[d.get((b, t), float("nan")) for t in steps] for b in blocks]
    return g, blocks, steps


def _heatmap(rows, key, title, path, cmap="magma", center=None):
    g, blocks, steps = _grid(rows, key)
    fig, ax = plt.subplots(figsize=(1.6 + 0.62 * len(steps),
                                    1.8 + 0.46 * len(blocks)))
    kw = {}
    if center is not None:
        vals = [v for row in g for v in row if v == v]
        m = max(abs(max(vals) - center), abs(center - min(vals)))
        kw = dict(vmin=center - m, vmax=center + m)
    im = ax.imshow(g, aspect="auto", cmap=cmap, **kw)
    ax.set_xticks(range(len(steps)), [str(t) for t in steps])
    ax.set_yticks(range(len(blocks)), [str(b) for b in blocks])
    ax.set_xlabel("denoising step index   (0 = noisiest, 49 = final)")
    ax.set_ylabel("DiT block index")
    ax.set_title(title, fontsize=11)
    for i in range(len(blocks)):
        for j in range(len(steps)):
            v = g[i][j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmap == "magma" else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def _depth_profile(rows, key, path):
    """The t=0 column is drift-free: both runs share the same initial latent,
    so any difference there is caused by injection alone, not trajectory drift."""
    g, blocks, steps = _grid(rows, key)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for t_pick in [0, 10, 20, 40]:
        if t_pick not in steps:
            continue
        j = steps.index(t_pick)
        ys = [g[i][j] for i in range(len(blocks))]
        style = dict(marker="o", linewidth=2.2) if t_pick == 0 else dict(
            marker=".", linewidth=1.2, alpha=0.65)
        label = f"step {t_pick}" + (" (drift-free)" if t_pick == 0 else "")
        ax.plot(blocks, ys, label=label, **style)
    ax.set_xlabel("DiT block index")
    ax.set_ylabel(key)
    ax.set_title("Injection effect vs depth\n"
                 "step 0 is the clean measurement: identical initial latent",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", type=str, required=True)
    args = p.parse_args()
    d = args.analysis_dir

    cross = _load(os.path.join(d, "cross_run_divergence.csv"))
    within = _load(os.path.join(d, "within_run_block_delta.csv"))

    _heatmap(cross, "cos_dist",
             "Cross-run: 1 - mean token cosine (injected vs baseline)\n"
             "higher = injection moved the representation further",
             os.path.join(d, "cross_run_cos_dist.png"))
    _heatmap(cross, "norm_ratio",
             "Cross-run: injected / baseline mean token norm\n"
             "1.0 = same magnitude; >1 = activations inflated",
             os.path.join(d, "cross_run_norm_ratio.png"),
             cmap="coolwarm", center=1.0)
    _heatmap(within, "rel_l2",
             "Within-run (baseline): relative L2 change vs previous captured block\n"
             "how much work each depth does - no cross-run drift confound",
             os.path.join(d, "within_run_rel_l2.png"))
    _depth_profile(cross, "cos_dist",
                   os.path.join(d, "depth_profile_cos_dist.png"))


if __name__ == "__main__":
    main()
