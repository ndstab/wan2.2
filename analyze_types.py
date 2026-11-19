"""Can the weather TYPE be read off Wan2.2's internal weather difference, and where?

For records with one weather factor each (paired_atlas.py --dataset image), at
every (block, sigma) cell:
  acc    leave-one-out nearest-centroid accuracy of the weather type from the
         normalised token-mean difference vector dmean. Centroids exclude the
         held-out sample. Chance = 1 / n_types.
  sep    mean cosine to same-type samples minus mean cosine to other types
         (pairs from the same source image never count).
and, at the best cell, the n_types x n_types mean-cosine matrix.

A training-free probe: no weights are fitted beyond class means.
CPU only; plots need matplotlib.
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = [torch.load(f, weights_only=False)
            for f in sorted(glob.glob(os.path.join(args.results, '*.pt')))]
    recs = [r for r in recs if r.get('effect')]
    types = np.array([r['effect'] for r in recs])
    src = np.array([r['shard'] for r in recs])
    names = sorted(set(types))
    N, K = len(recs), len(names)
    S = len(recs[0]['sigmas'])
    B = 1 + max(b for b, _ in recs[0]['blocks'])
    sig = recs[0]['sigmas']
    print(f"{N} pairs, types {dict(Counter(types))}")

    acc = np.zeros((B, S)); sep = np.zeros((B, S))
    same = types[:, None] == types[None, :]
    other_src = src[:, None] != src[None, :]
    for b in range(B):
        for s in range(S):
            V = torch.stack([r['blocks'][(b, s)]['dmean'].float() for r in recs]).numpy()
            V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
            C = V @ V.T
            sep[b, s] = C[same & other_src].mean() - C[~same & other_src].mean()
            correct = 0
            for i in range(N):
                keep = other_src[i]
                cents = []
                for t in names:
                    m = keep & (types == t)
                    c = V[m].mean(0) if m.any() else np.zeros(V.shape[1])
                    cents.append(c / (np.linalg.norm(c) + 1e-8))
                correct += names[int(np.argmax(np.array(cents) @ V[i]))] == types[i]
            acc[b, s] = correct / N

    bb, ss = np.unravel_index(np.argmax(acc + 1e-3 * sep), acc.shape)
    V = torch.stack([r['blocks'][(bb, ss)]['dmean'].float() for r in recs]).numpy()
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    C = V @ V.T
    M = np.zeros((K, K))
    for i, a in enumerate(names):
        for j, c in enumerate(names):
            m = (types[:, None] == a) & (types[None, :] == c) & other_src
            M[i, j] = C[m].mean() if m.any() else np.nan

    print(f"chance accuracy {1 / K:.3f}")
    print(f"best cell: block {bb}, sigma {sig[ss]:.3f}: accuracy {acc[bb, ss]:.3f}, "
          f"separation {sep[bb, ss]:.3f}")
    print("best block per sigma (accuracy):")
    for s in range(S):
        b = int(acc[:, s].argmax())
        print(f"  sigma {sig[s]:.3f}: block {b:2d} acc {acc[b, s]:.3f} "
              f"(blocks 19-22 mean {acc[19:23, s].mean():.3f}, block 0 {acc[0, s]:.3f})")
    print("type x type mean cosine at the best cell:")
    print("          " + " ".join(f"{n[:8]:>8s}" for n in names))
    for i, a in enumerate(names):
        print(f"{a[:8]:>9s} " + " ".join(f"{M[i, j]:8.3f}" for j in range(K)))
    torch.save(dict(acc=acc, sep=sep, sigmas=sig, names=names, M=M, best=(int(bb), int(ss)),
                    counts=dict(Counter(types))), os.path.join(args.out, 'types_summary.pt'))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for a, Mx, t, cm in [(ax[0], acc, f'weather-type accuracy (chance {1 / K:.2f})', 'viridis'),
                         (ax[1], sep, 'same-type minus other-type cosine', 'magma')]:
        im = a.imshow(Mx, aspect='auto', origin='lower', cmap=cm)
        a.set_xticks(range(S)); a.set_xticklabels([f"{x:.2f}" for x in sig], rotation=45)
        a.set(xlabel='noise level sigma (noisy -> clean)', ylabel='DiT block', title=t)
        fig.colorbar(im, ax=a, shrink=0.85)
    im = ax[2].imshow(M, cmap='coolwarm', vmin=-1, vmax=1)
    ax[2].set_xticks(range(K)); ax[2].set_xticklabels(names, rotation=45)
    ax[2].set_yticks(range(K)); ax[2].set_yticklabels(names)
    ax[2].set_title(f'type similarity, block {bb}, sigma {sig[ss]:.2f}')
    for i in range(K):
        for j in range(K):
            ax[2].text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax[2], shrink=0.85)
    fig.suptitle(f'Is the weather type separable inside Wan2.2? {N} single-factor image pairs',
                 x=0.01, ha='left')
    p = os.path.join(args.out, 'fig_types.png')
    fig.savefig(p, dpi=150)
    print('wrote', p)


if __name__ == '__main__':
    main()
