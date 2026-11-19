"""Summarise paired_inversion.py records (T5.3b) and plot them.

Questions answered:
  1. Along the deterministic inversion of a clear video and its weather twin,
     at which noise level does the latent weather difference vanish?
     (lat_ratio(sigma); half-life sigma per pair)
  2. Do the two videos invert to the same noise? (noise_cos)
  3. How faithful is the inversion? (recon_rel; recon_diff_ratio = does
     regeneration from the inverted noises restore the weather difference)
  4. Inside the network, where (block) and when (sigma) is the weather
     difference largest along the inversion trajectory? (rel[b, sigma])
  5. Optional (--atlas): does the inversion-trajectory weather direction agree
     with the forward-noising atlas direction at the same block / sigma?

CPU only. Plots need matplotlib; the summary does not.
"""
import argparse
import glob
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--atlas', default=None, help='paired_atlas.py --out dir (same pairs)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = [torch.load(f, weights_only=False)
            for f in sorted(glob.glob(os.path.join(args.results, '*.pt')))]
    N = len(recs)
    grid = recs[0]['grid']
    probe = recs[0]['probe']
    sig = np.array(grid[1:])                         # sigma after each step
    lat = np.array([r['lat_ratio'] for r in recs])   # [N, steps]
    lcos = np.array([r['lat_cos'] for r in recs])
    ncos = np.array([r['noise_cos'] for r in recs])
    B = 1 + max(b for b, _ in recs[0]['blocks'])
    rel = np.array([[[r['blocks'][(b, i)]['rel'] for i in probe] for b in range(B)]
                    for r in recs])                  # [N, B, P]
    probe_sig = np.array([max(grid[i], grid[1]) for i in probe])

    def half_life(row, frac):
        j = next((k for k, v in enumerate(row) if v < frac), None)
        return sig[j] if j is not None else np.nan

    hl50 = np.array([half_life(r, 0.5) for r in lat])
    hl20 = np.array([half_life(r, 0.2) for r in lat])
    labels = [f"{r['shard']}_{r['variant']} " + ",".join(
        f"{k}{v:g}" for k, v in r['params'].items() if v) for r in recs]

    print(f"{N} pairs, {len(sig)} inversion steps, sigma {sig[0]:.4f}..{sig[-1]:.4f}")
    print("\nlatent weather difference remaining along the inversion (mean over pairs)")
    for s_target in (0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, sig[-1]):
        j = int(np.argmin(np.abs(sig - s_target)))
        print(f"  sigma {sig[j]:.3f}: ratio {lat[:, j].mean():.3f} +- {lat[:, j].std():.3f}"
              f"   cos-to-original {lcos[:, j].mean():.3f}")
    print(f"\nsigma where half of the difference is gone: {np.nanmean(hl50):.3f} "
          f"(per pair {np.round(hl50, 3).tolist()})")
    print(f"sigma where 80% is gone: {np.nanmean(hl20):.3f}")
    print(f"cosine between the two inverted noises: {ncos.mean():.3f} +- {ncos.std():.3f}")
    if 'recon_rel' in recs[0]:
        rr = np.array([r['recon_rel'] for r in recs])
        rd = np.array([r['recon_diff_ratio'] for r in recs])
        print(f"reconstruction rel-L2 clear {rr[:, 0].mean():.3f}, weather {rr[:, 1].mean():.3f}; "
              f"weather difference restored after regeneration: {rd.mean():.3f}")
    rm = rel.mean(0)
    print("\nper-block |dh|/|h| along the inversion (mean), peak block per sigma")
    for p, s in enumerate(probe_sig):
        b = int(rm[:, p].argmax())
        print(f"  sigma {s:.3f}: peak block {b:2d} ({rm[b, p]:.3f}), "
              f"block 0 {rm[0, p]:.3f}, block 29 {rm[-1, p]:.3f}")

    summary = dict(sig=sig, lat=lat, lcos=lcos, ncos=ncos, hl50=hl50, hl20=hl20,
                   rel=rel, probe_sig=probe_sig, labels=labels)

    if args.atlas:
        arecs = {f"{r['shard']}_{r['variant']}": r for r in
                 (torch.load(f, weights_only=False)
                  for f in glob.glob(os.path.join(args.atlas, '*.pt')))}
        asig = None
        agree = []
        for r in recs:
            a = arecs.get(f"{r['shard']}_{r['variant']}")
            if a is None:
                continue
            asig = np.array(a['sigmas'])
            row = np.full((B, len(asig)), np.nan)
            for s_i, s in enumerate(asig):
                p = int(np.argmin(np.abs(probe_sig - s)))
                for b in range(B):
                    u = r['blocks'][(b, probe[p])]['dmean'].float()
                    v = a['blocks'][(b, s_i)]['dmean'].float()
                    row[b, s_i] = torch.nn.functional.cosine_similarity(u, v, dim=0).item()
            agree.append(row)
        if agree:
            agree = np.array(agree)
            summary.update(atlas_agree=agree, atlas_sig=asig)
            m = np.nanmean(agree, 0)
            print("\ninversion vs forward-noising direction (cosine, mean over pairs)")
            for s_i, s in enumerate(asig):
                print(f"  sigma {s:.3f}: blocks 19-22 {np.nanmean(m[19:23, s_i]):.3f}  "
                      f"all {np.nanmean(m[:, s_i]):.3f}")

    torch.save(summary, os.path.join(args.out, 'inversion_summary.pt'))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped plots")
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for row, lab in zip(lat, labels):
        ax[0].plot(sig, row, alpha=0.35, lw=1)
    ax[0].plot(sig, lat.mean(0), 'k', lw=2.5, label='mean')
    ax[0].axvspan(0.83, 0.98, color='tab:orange', alpha=0.12,
                  label='forward-noising weather window')
    ax[0].set(xlabel='noise level sigma (inversion runs left to right)',
              ylabel='|x_w - x_c| / |x_w(0) - x_c(0)|',
              title='How much of the weather difference survives inversion')
    ax[0].legend(frameon=False)
    ax[1].plot(sig, lcos.mean(0), 'k', lw=2)
    ax[1].fill_between(sig, lcos.mean(0) - lcos.std(0), lcos.mean(0) + lcos.std(0),
                       color='k', alpha=0.1)
    ax[1].set(xlabel='sigma', ylabel='cosine to the clean-image difference',
              title='Is it still the same difference?')
    im = ax[2].imshow(rm, aspect='auto', origin='lower', cmap='magma',
                      extent=[0, len(probe_sig), -0.5, B - 0.5])
    ax[2].set_xticks(np.arange(len(probe_sig)) + 0.5)
    ax[2].set_xticklabels([f"{s:.2f}" for s in probe_sig], rotation=60, fontsize=7)
    ax[2].set(xlabel='sigma along the inversion', ylabel='block',
              title='|dh| / |h| inside the DiT')
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout()
    p = os.path.join(args.out, 'fig_inversion.png')
    plt.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == '__main__':
    main()
