"""T5.4 -- build weather steering directions from paired-atlas records.

CPU only (runs on the login node). Reads the per-pair records written by
paired_atlas.py, averages the token-mean residual difference per
(block, sigma), and writes a directions file for --steer_vectors.

Also prints a leave-one-shard-out stability check: the cosine between the
direction built without each shard and the full direction. If that is low,
the direction is dominated by a few scenes and should not be trusted.

Examples:
  python build_steer_vectors.py --records t5/results/scene0 --out t5/steer/all.pt
  python build_steer_vectors.py --records t5/results/scene0 --out t5/steer/fog.pt \
      --filter 'fog>=5'
"""
import argparse
import os

import torch

from wan.utils.steering import build_directions, load_records, parse_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True, help='paired_atlas.py --out dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--filter', default=None, help="e.g. 'fog>=5,rain==0'")
    ap.add_argument('--exclude_shards', default='', help='comma-separated shard ids')
    ap.add_argument('--report_blocks', default='19,20,21,22')
    args = ap.parse_args()

    recs = load_records(args.records)
    excl = [s for s in args.exclude_shards.split(',') if s]
    d = build_directions(recs, filt=args.filter, exclude_shards=excl)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(d, args.out)

    blocks = [int(b) for b in args.report_blocks.split(',')]
    print(f"{d['n_pairs']} pairs -> {args.out}  dirs {tuple(d['dirs'].shape)}")
    print("sigma   " + "".join(f"  b{b:<6d}" for b in blocks) + "  (|direction|)")
    for s_i, s in enumerate(d['sigmas']):
        print(f"{s:.3f}  " + "".join(f"  {d['dirs'][b, s_i].norm():7.2f}" for b in blocks))

    pred = parse_filter(args.filter)
    shards = sorted({r['shard'] for r in recs if pred(r['params'])} - set(excl))
    if len(shards) < 3:
        return
    print("\nleave-one-shard-out cosine to full direction (mean over report blocks)")
    print("shard   " + "".join(f"  {s:.2f}" for s in d['sigmas']))
    full = d['dirs'][blocks]
    for sh in shards:
        loo = build_directions(recs, filt=args.filter,
                               exclude_shards=excl + [sh])['dirs'][blocks]
        cos = torch.nn.functional.cosine_similarity(loo, full, dim=-1).mean(0)
        print(f"{sh}  " + "".join(f"  {c:.2f}" for c in cos.tolist()))


if __name__ == '__main__':
    main()
