"""Aggregate paired_atlas.py outputs into one small summary file.

Answers, from the per-pair records:
  T5.1  where -- block x sigma map of the weather difference, raw and
        normalised by block 0 (block 0 tracks the input difference, so the
        ratio is how much the network itself amplifies or suppresses weather);
  T5.2  disentanglement -- is the token-mean difference vector consistent
        ACROSS scenes? Cross-scene cosine within a weather type vs across
        types. A content-independent weather direction shows up as high
        cross-scene cosine at some (block, sigma);
  T5.3  phase -- how much of the true weather difference survives into the
        model's x0 prediction at each sigma.

CPU only. Writes <out>/summary.pt and <out>/summary.csv.
"""
import argparse
import glob
import os

import numpy as np
import torch


def wtype(params):
    if all(v == 0 for v in params.values()):
        return 'content'          # control pair: two clear scenes
    return 'snowflake' if params.get('snowflake', 0) > 0 else 'rain'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.results, '*.pt')))
    recs = [torch.load(f, weights_only=False) for f in files]
    N, B, S = len(recs), 30, len(recs[0]['sigmas'])
    sigmas = recs[0]['sigmas']
    print(f"{N} pairs, {B} blocks, {S} sigmas")

    rel = np.zeros((N, B, S))
    cos_tok = np.zeros((N, B, S))
    D = np.zeros((N, B, S, recs[0]['blocks'][(0, 0)]['dmean'].numel()), dtype=np.float32)
    x0_ratio = np.zeros((N, S))
    x0_cos = np.zeros((N, S))
    shards, types, params = [], [], []
    for n, r in enumerate(recs):
        shards.append(r['shard']); types.append(r.get('effect') or wtype(r['params'])); params.append(r['params'])
        for b in range(B):
            for s in range(S):
                st = r['blocks'][(b, s)]
                rel[n, b, s] = st['norm_d'] / st['norm_c']
                cos_tok[n, b, s] = st['cos_tok']
                D[n, b, s] = st['dmean'].float().numpy()
        for s in range(S):
            x0_ratio[n, s] = r['x0pred'][s]['pred_over_true']
            x0_cos[n, s] = r['x0pred'][s]['cos_pred_true']

    amp = rel / rel[:, :1, :]

    # ---- cross-scene consistency of the token-mean difference direction ----
    shards_a = np.array(shards); types_a = np.array(types)
    diff_scene = shards_a[:, None] != shards_a[None, :]
    same_type = types_a[:, None] == types_a[None, :]
    iu = np.triu(np.ones((N, N), bool), 1)
    m_within = iu & diff_scene & same_type          # same weather type, different scene
    m_across = iu & diff_scene & ~same_type         # different weather type, different scene
    cons_within = np.zeros((B, S)); cons_across = np.zeros((B, S))
    for b in range(B):
        for s in range(S):
            V = D[:, b, s]
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
            C = V @ V.T
            cons_within[b, s] = C[m_within].mean() if m_within.any() else np.nan
            cons_across[b, s] = C[m_across].mean() if m_across.any() else np.nan

    # random-direction reference for a 3072-d cosine
    rng = np.random.default_rng(0)
    R = rng.standard_normal((200, D.shape[-1])); R /= np.linalg.norm(R, axis=1, keepdims=True)
    rand_cos_sd = (R @ R.T)[np.triu(np.ones((200, 200), bool), 1)].std()

    summary = dict(
        sigmas=sigmas, shards=shards, types=types, params=params,
        rel_mean=rel.mean(0), rel_sd=rel.std(0),
        amp_mean=amp.mean(0), amp_sd=amp.std(0),
        cos_tok_mean=cos_tok.mean(0),
        cons_within=cons_within, cons_across=cons_across,
        n_within=int(m_within.sum()), n_across=int(m_across.sum()),
        rand_cos_sd=float(rand_cos_sd),
        x0_ratio_mean=x0_ratio.mean(0), x0_ratio_sd=x0_ratio.std(0),
        x0_cos_mean=x0_cos.mean(0), x0_cos_sd=x0_cos.std(0),
        # keep token maps for the first shard only (for spatial figures)
        tokmaps={(r['shard'], r['variant'], b, s): r['blocks'][(b, s)]['tokmap']
                 for r in recs if r['shard'] == shards[0]
                 for b in range(B) for s in range(S)},
        tok_shape=recs[0]['tok_shape'],
    )
    torch.save(summary, os.path.join(args.out, 'summary.pt'))

    with open(os.path.join(args.out, 'summary.csv'), 'w') as f:
        f.write('block,sigma,rel_mean,amp_mean,cos_tok_mean,cons_within,cons_across\n')
        for b in range(B):
            for s in range(S):
                f.write(f"{b},{sigmas[s]:.4f},{summary['rel_mean'][b, s]:.5f},"
                        f"{summary['amp_mean'][b, s]:.4f},{summary['cos_tok_mean'][b, s]:.5f},"
                        f"{cons_within[b, s]:.4f},{cons_across[b, s]:.4f}\n")

    # ---- console digest ----
    print(f"\nrandom 3072-d cosine sd = {rand_cos_sd:.4f}  (within pairs={summary['n_within']}, across={summary['n_across']})")
    print("\nsigma   peak-amp block   amp@peak   x0 kept   x0 cos   best cons_within (block)   cons_across there")
    for s in range(S):
        bp = int(summary['amp_mean'][:, s].argmax())
        col = cons_within[:, s]
        bc = int(np.nanargmax(col)) if np.isfinite(col).any() else 0
        print(f"{sigmas[s]:.3f}   {bp:>14d}   {summary['amp_mean'][bp, s]:8.2f}   "
              f"{summary['x0_ratio_mean'][s]:7.3f}   {summary['x0_cos_mean'][s]:6.3f}   "
              f"{cons_within[bc, s]:8.3f} (b{bc:02d})          {cons_across[bc, s]:8.3f}")
    print("ANALYZE_DONE")


if __name__ == '__main__':
    main()
