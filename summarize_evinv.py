"""Aggregate paired_eval_inv.py metrics.csv files into a per-arm table."""
import collections
import csv
import os
import sys

R = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/wan/results")
for d in ["evinv_scene0", "evinv_scene0_knn", "evinv_knn_early", "evinv_knn_all"]:
    p = f"{R}/{d}/metrics.csv"
    if not os.path.exists(p):
        continue
    rows = [r for r in csv.DictReader(open(p)) if r.get("arm") and r["arm"] != "arm"]
    agg = collections.defaultdict(list)
    pairs = set()
    for r in rows:
        pairs.add((r["shard"], r["variant"]))
        for k in ("frac", "cos", "err", "psnr_weather", "psnr_clear"):
            if r.get(k):
                agg[(r["arm"], k)].append(float(r[k]))
    print(f"\n== {d}: {len(pairs)} pairs, shards {sorted({s for s, _ in pairs})}")
    arms = sorted({a for a, _ in agg}, key=lambda a: (a != "base", a))
    print(f"{'arm':22s} frac   cos    err    psnrW  psnrC  n   err<1")
    for a in arms:
        def m(k):
            v = agg[(a, k)]
            return sum(v) / len(v) if v else float("nan")
        errs = agg[(a, "err")]
        print(f"{a:22s} {m('frac'):.2f}  {m('cos'):.2f}  {m('err'):.2f}  "
              f"{m('psnr_weather'):5.1f}  {m('psnr_clear'):5.1f}  {len(errs):2d}  "
              f"{sum(e < 1.0 for e in errs)}")
