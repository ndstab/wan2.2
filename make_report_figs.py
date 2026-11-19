"""Figures for the course report (report/figs/). Runs locally, CPU only.

New figures built here:
  fig_phase_worlds.pdf   x0-retention of the weather vs the scene difference,
                         one panel per Unreal world + the SDXL image set
  fig_cross_env.pdf      cross-environment cosine of the weather direction
  fig_steer_sweep.jpg    labelled text-to-video steering sweep
  fig_dataset.jpg        one WeatherWeaver simulation pair + image-set samples
Other figures are copied from outputs/ unchanged.

The per-world numbers are the analyze_paired.py outputs recorded on 2026-10-19
(prajna run, logged in CLAUDE.md); the cross-environment matrix is
outputs/t5_prajna/summary_cross_env.pt.
"""
import os
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'report', 'figs')
O = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42,
})

SIG = [0.999, 0.978, 0.952, 0.920, 0.882, 0.833, 0.768, 0.681, 0.555, 0.356, 0.092]
WORLDS = {  # name: (weather x0-retention, content-control x0-retention)
    'City (scene 0)': ([.009, .231, .415, .615, .664, .703, .733, .752, .786, .874, .978],
                       [.006, .111, .188, .286, .373, .508, .626, .729, .802, .879, .976]),
    'Town (scene 1)': ([.011, .213, .384, .516, .617, .658, .677, .706, .755, .848, .974],
                       [.007, .152, .295, .491, .647, .738, .794, .841, .883, .933, .990]),
    'Bush road (scene 2)': ([.010, .233, .354, .521, .564, .594, .608, .641, .684, .783, .968],
                            [.005, .096, .220, .433, .546, .636, .677, .726, .776, .855, .979]),
    'Rural (scene 3)': ([.009, .269, .469, .624, .704, .750, .778, .806, .839, .899, .982],
                        [.005, .102, .257, .442, .559, .656, .722, .783, .838, .904, .983]),
}
IMAGES = [.011, .207, .330, .423, .461, .491, .526, .576, .641, .744, .931]


def fig_phase_worlds():
    # Evenly spaced sigma ticks (the probed steps), so the high-noise window
    # where the curves separate is not squashed against the left edge.
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7), sharey=True,
                             constrained_layout=True)
    x = np.arange(len(SIG))
    win = [i for i, s in enumerate(SIG) if 0.9 <= s <= 0.98]

    def style(ax, title):
        ax.axvspan(min(win) - 0.5, max(win) + 0.5, color=GRID, alpha=0.8, lw=0)
        ax.set_title(title, loc='left')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{s:.2f}" for s in SIG[::2]], fontsize=7)
        ax.grid(axis='y', color=GRID, lw=0.6)

    for ax, (name, (w, c)) in zip(axes, WORLDS.items()):
        style(ax, name)
        ax.plot(x, w, color=BLUE, lw=2, marker='o', ms=3.5, label='weather difference')
        ax.plot(x, c, color=ORANGE, lw=2, marker='o', ms=3.5,
                label='scene difference (control)')
    style(axes[4], 'SDXL image pairs (weather only)')
    axes[4].plot(x, IMAGES, color=BLUE, lw=2, marker='o', ms=3.5)
    axes[0].set_ylabel('fraction of the difference\ncommitted in $\\hat{x}_0$')
    axes[0].annotate('weather', (2, WORLDS['City (scene 0)'][0][2]), (2.6, 0.62),
                     fontsize=8, color=INK)
    axes[0].annotate('scene', (2, WORLDS['City (scene 0)'][1][2]), (3.2, 0.12),
                     fontsize=8, color=INK)
    fig.supxlabel('noise level $\\sigma$ at the probed step (generation runs left to right; '
                  'shaded = $\\sigma$ 0.92-0.98)', fontsize=8.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=2, bbox_to_anchor=(1.0, 1.12))
    fig.savefig(os.path.join(OUT, 'fig_phase_worlds.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_cross_env():
    S = torch.load(os.path.join(O, 't5_prajna', 'summary_cross_env.pt'), weights_only=False)
    sig = np.array(S['sigmas'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    for key, col, lab in [('all', BLUE, 'all weather pairs'),
                          ('same', AQUA, 'same weather family'),
                          ('diff', YELLOW, 'different family'),
                          ('control', ORANGE, 'control: two clear scenes')]:
        y = S[key][19:23].mean(0)
        ax.plot(sig, y, color=col, lw=2, marker='o', ms=3.5, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(1.02, 0.05)
    ax.set_xlabel('noise level $\\sigma$')
    ax.set_ylabel('cosine across different worlds')
    ax.set_title('Blocks 19-22: is the weather direction shared across worlds?', loc='left')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[1]
    im = ax.imshow(S['all'], aspect='auto', origin='lower', cmap='Blues',
                   vmin=0, vmax=max(0.6, S['all'].max()))
    ax.set_xticks(range(len(sig)))
    ax.set_xticklabels([f"{s:.2f}" for s in sig], rotation=60, fontsize=7)
    ax.set_xlabel('noise level $\\sigma$ (noisy $\\rightarrow$ clean)')
    ax.set_ylabel('DiT block')
    ax.set_title('All weather pairs, every block', loc='left')
    fig.colorbar(im, ax=ax, shrink=0.85, label='mean cosine')
    fig.savefig(os.path.join(OUT, 'fig_cross_env.pdf'), bbox_inches='tight')
    plt.close(fig)


def _font(sz):
    return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', sz)


def labelled_grid(tiles, ncol, out, w=400, bar=28):
    """tiles: [(png_path, label)] -> one labelled jpg."""
    ims = [Image.open(p).convert('RGB') for p, _ in tiles]
    h = round(ims[0].height * w / ims[0].width)
    nrow = (len(tiles) + ncol - 1) // ncol
    sheet = Image.new('RGB', (ncol * w, nrow * (h + bar)), 'white')
    d = ImageDraw.Draw(sheet)
    for i, (im, (_, lab)) in enumerate(zip(ims, tiles)):
        r, c = divmod(i, ncol)
        x, y = c * w, r * (h + bar)
        d.text((x + 6, y + 5), lab, fill=(20, 20, 20), font=_font(16))
        sheet.paste(im.resize((w, h), Image.LANCZOS), (x, y + bar))
    sheet.save(out, quality=90)


def fig_steer_sweep():
    g = os.path.join(O, 't5_prajna', 'steer_gen')
    tiles = [('base_p1', 'no steering'), ('a1_p1', 'alpha=1'), ('a2_p1', 'alpha=2'),
             ('a4_p1', 'alpha=4'), ('a8_p1', 'alpha=8'),
             ('a2wide_p1', 'alpha=2, wide window'), ('a2cond_p1', 'alpha=2, cond. pass only'),
             ('a4norm_p1', 'alpha=4, norm-preserving'), ('base_p2', 'dashcam, no steering'),
             ('a4_p2', 'dashcam, alpha=4')]
    labelled_grid([(os.path.join(g, f't_{n}.png'), l) for n, l in tiles], 5,
                  os.path.join(OUT, 'fig_steer_sweep.jpg'))


def fig_dataset():
    """Clear vs weather frame of one simulation pair + the image-set sheet."""
    d = os.path.expanduser('~/Downloads/0000')
    tiles = []
    labels = {'0000': 'clear (all parameters 0)', '0001': 'cloud5 rain9 fog1',
              '0003': 'cloud7 rain7 fog10 puddle1', '0004': 'cloud3 fog8 snowflake1'}
    tmp = os.path.join(OUT, '_tmp')
    os.makedirs(tmp, exist_ok=True)
    for v, lab in labels.items():
        png = os.path.join(tmp, f'{v}.png')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(d, f'{v}.mp4'),
                        '-vf', 'select=eq(n\\,50),crop=1920:1056:0:112,scale=640:-2',
                        '-vframes', '1', png], check=True)
        tiles.append((png, lab))
    labelled_grid(tiles, 4, os.path.join(OUT, 'fig_dataset_sim.jpg'))
    shutil.rmtree(tmp)


COPY = {
    'latent-inspection/analysis/lambda_sweep_full.jpg': 'fig_lambda_sweep.jpg',
    'latent-inspection/analysis/block_subsets.jpg': 'fig_block_subsets.jpg',
    'latent-inspection/analysis/prompt_generality.jpg': 'fig_prompt_generality.jpg',
    'latent-inspection/analysis/depth_step0_only.png': 'fig_atlas_step0.png',
    'gen_inv/sheet_matched_ladder.png': 'fig_inv_ladder.png',
    'gen_inv/pareto_inverted_vs_random.png': 'fig_inv_pareto.png',
    't5_paired/figs/fig_where.png': 'fig_where.png',
    't5_paired/figs/fig_consistency.png': 'fig_consistency.png',
    't5_paired/figs/fig_spatial.png': 'fig_spatial.png',
    't5_prajna/summary_image/fig_types.png': 'fig_types.png',
    't5_prajna/summary_inversion_scene0_fp3/fig_inversion.png': 'fig_inversion.png',
    't5_prajna/gated_gen/gated_sheet.jpg': 'fig_gated_sheet.jpg',
    't5_paired/ww_image_samples/weatherweaver_image_samples.jpg': 'fig_dataset_images.jpg',
    'assets/ref_strip.jpg': 'fig_sandstorm_ref.jpg',
}

if __name__ == '__main__':
    for src, dst in COPY.items():
        shutil.copy(os.path.join(O, src), os.path.join(OUT, dst))
    fig_phase_worlds()
    fig_cross_env()
    fig_steer_sweep()
    fig_dataset()
    print('figures in', OUT)
    for f in sorted(os.listdir(OUT)):
        print('  ', f)
