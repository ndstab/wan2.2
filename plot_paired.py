"""Figures for the T5 paired-difference atlas. Runs locally (cluster has no matplotlib).

  fig_where.png      T5.1  block x sigma: how much the network amplifies the
                           weather difference relative to block 0 (the input)
  fig_consistency.png T5.2 cross-scene cosine of the weather difference
                           direction: same type / different type / content control
  fig_phase.png      T5.3  how much of the true weather difference survives
                           into the model's x0 prediction, vs sigma
  fig_spatial.png    T5.1  where in the frame the difference sits, one pair

Palette: single-hue blue ramp for magnitude; blue<->red diverging with a grey
midpoint for signed cosines. Text in ink, never in series colour.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import torch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])
S1, S2 = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})


def sigma_ticks(ax, sigmas):
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:.2f}" for s in sigmas], rotation=45)
    ax.set_xlabel("noise level σ   (noisy → clean)")


def heat(ax, M, sigmas, cmap, norm=None, vmin=None, vmax=None, title=""):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    sigma_ticks(ax, sigmas)
    ax.set_yticks(range(0, 30, 5))
    ax.set_ylabel("DiT block")
    ax.set_title(title, loc="left")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--control", help="summary.pt from --pair_mode content")
    ap.add_argument("--frames", help="dir of local shard-0000 mp4s for the spatial figure")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = torch.load(args.summary, weights_only=False)
    sig = S["sigmas"]
    n_pairs = len(S["shards"])

    # ---- T5.1 where ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    im = heat(axes[0], S["rel_mean"], sig, SEQ,
              title="Weather difference, ‖Δh‖ / ‖h_clear‖")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    im = heat(axes[1], S["amp_mean"], sig, SEQ,
              title="Amplification by the network (÷ block 0)")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.suptitle(f"Where Wan2.2 represents weather — mean of {n_pairs} pixel-aligned "
                 f"clear/weather pairs, shared noise, null prompt", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.savefig(os.path.join(args.out, "fig_where.png"), dpi=150)
    plt.close(fig)

    # ---- T5.2 consistency ----------------------------------------------
    panels = [(S["cons_within"], f"Same weather type, different scene  (n={S['n_within']})"),
              (S["cons_across"], f"Different weather type, different scene  (n={S['n_across']})")]
    if args.control:
        C = torch.load(args.control, weights_only=False)
        panels.append((C["cons_within"],
                       f"CONTROL: clear scene vs other clear scene  (n={C['n_within']})"))
    lim = max(np.nanmax(np.abs(p[0])) for p in panels)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6),
                             constrained_layout=True)
    for ax, (M, t) in zip(axes, panels):
        im = heat(ax, M, sig, DIV, norm=norm, title=t)
    fig.colorbar(im, ax=axes, shrink=0.85, label="mean cosine of token-mean Δh")
    fig.suptitle("Is there a content-independent weather direction?  "
                 f"Random 3072-d vectors: cosine ≈ 0 ± {S['rand_cos_sd']:.3f}",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(os.path.join(args.out, "fig_consistency.png"), dpi=150)
    plt.close(fig)

    # ---- T5.3 phase ------------------------------------------------------
    x = np.arange(len(sig))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, key, title in [
            (axes[0], "x0_ratio", "Fraction of the true difference kept in x̂₀  (‖Δx̂₀‖ / ‖Δx₀‖)"),
            (axes[1], "x0_cos", "Direction agreement, cos(Δx̂₀, Δx₀)")]:
        series = [(S, S1, f"weather pairs (n={n_pairs})")]
        if args.control:
            series.append((C, S2, f"control: different clear scenes (n={len(C['shards'])})"))
        for src, col, lab in series:
            m, sd = src[key + "_mean"], src[key + "_sd"]
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.13, linewidth=0)
            ax.plot(x, m, color=col, linewidth=2, marker="o", markersize=5, label=lab)
        ax.legend(frameon=False, loc="lower right", labelcolor=INK)
        sigma_ticks(ax, sig)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylim(bottom=0)
    fig.suptitle("When does the model see the weather vs the scene?  Band = ±1 sd",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(os.path.join(args.out, "fig_phase.png"), dpi=150)
    plt.close(fig)

    # ---- spatial ---------------------------------------------------------
    if args.frames:
        import subprocess
        tok = S["tokmaps"]
        T, h, w = S["tok_shape"]
        shard0 = S["shards"][0]
        variants = sorted({k[1] for k in tok if k[0] == shard0})[:2]
        s_pick = int(np.argmax(S["x0_cos_mean"] > 0.5)) if (S["x0_cos_mean"] > 0.5).any() else len(sig) // 2
        b_pick = int(S["amp_mean"][:, s_pick].argmax())
        fig, axes = plt.subplots(len(variants), 3, figsize=(13, 3.0 * len(variants)),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for r, v in enumerate(variants):
            frames = []
            for vid in ("0000", v):
                p = os.path.join(args.frames, f"{vid}.mp4")
                png = os.path.join(args.out, f"_f_{vid}.png")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-vf",
                                "select=eq(n\\,50),crop=1920:1056:0:112,scale=640:352",
                                "-vframes", "1", png], check=True)
                frames.append(plt.imread(png))
            m = tok[(shard0, v, b_pick, s_pick)].float().numpy().reshape(T, h, w).mean(0)
            axes[r, 0].imshow(frames[0]); axes[r, 0].set_title("clear", loc="left")
            axes[r, 1].imshow(frames[1]); axes[r, 1].set_title(f"weather variant {v}", loc="left")
            # One corner token carries an extreme value that would set the whole
            # colour scale; clip the scale at the 98th percentile for display.
            im = axes[r, 2].imshow(m, cmap=SEQ, interpolation="bilinear", aspect="auto",
                                   vmin=np.percentile(m, 2), vmax=np.percentile(m, 98))
            axes[r, 2].set_title(f"‖Δh‖ per token, block {b_pick}, σ={sig[s_pick]:.2f}", loc="left")
            fig.colorbar(im, ax=axes[r, 2], shrink=0.85)
            for a in axes[r]:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle("Where in the frame the network registers the weather (time-averaged; colour scale clipped at 2nd/98th percentile)",
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.savefig(os.path.join(args.out, "fig_spatial.png"), dpi=150)
        plt.close(fig)
        for f in os.listdir(args.out):
            if f.startswith("_f_"):
                os.remove(os.path.join(args.out, f))
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
