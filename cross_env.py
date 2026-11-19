"""Cross-environment consistency of the weather direction (T5.2, 4 Unreal worlds).

cos(dmean_i, dmean_j) for weather pairs i, j from DIFFERENT environments, per
(block, sigma); content-control pairs (clear vs other clear) give the null.
Reads paired_atlas.py records from <results>/atlas_scene{0..3}[_content].
"""
import glob
import os
import sys

import numpy as np
import torch

R = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/wan/results")


def load(pattern, tag):
    out = []
    for f in sorted(glob.glob(pattern)):
        r = torch.load(f, weights_only=False)
        out.append(dict(env=tag, params=r["params"], sig=r["sigmas"],
                        D={k: v["dmean"].float() for k, v in r["blocks"].items()}))
    return out


W = sum([load(f"{R}/atlas_scene{s}/*.pt", s) for s in range(4)], [])
C = sum([load(f"{R}/atlas_scene{s}_content/*.pt", s) for s in range(4)], [])
sig = W[0]["sig"]
B, S = 30, len(sig)


def fam(p):
    if p.get("snowflake", 0) > 0:
        return "snowflake"
    return "snow" if p.get("snow", 0) > 0 else "rain/fog/cloud"


def cross(recs, same_family=None):
    env = np.array([r["env"] for r in recs])
    fa = np.array([fam(r["params"]) for r in recs])
    m = np.triu(env[:, None] != env[None, :], 1)
    if same_family is True:
        m &= fa[:, None] == fa[None, :]
    if same_family is False:
        m &= fa[:, None] != fa[None, :]
    M = np.zeros((B, S))
    for b in range(B):
        for s in range(S):
            V = torch.stack([r["D"][(b, s)] for r in recs]).numpy()
            V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
            M[b, s] = (V @ V.T)[m].mean()
    return M, int(m.sum())


print("pairs per env:", {s: sum(r["env"] == s for r in W) for s in range(4)},
      "controls:", {s: sum(r["env"] == s for r in C) for s in range(4)})
A, nA = cross(W)
Sm, nS = cross(W, True)
Df, nD = cross(W, False)
Cc, nC = cross(C)
print(f"cross-environment cosine (n: all {nA}, same family {nS}, diff family {nD}, control {nC})")
print("sigma   all(b19-22 / best block)   same-family   diff-family   CONTROL")
for s in range(S):
    b = int(A[:, s].argmax())
    print(f"{sig[s]:.3f}   {A[19:23, s].mean():6.3f} / {A[b, s]:.3f} (b{b:02d})   "
          f"{Sm[19:23, s].mean():6.3f}   {Df[19:23, s].mean():6.3f}   {Cc[19:23, s].mean():6.3f}")
torch.save(dict(all=A, same=Sm, diff=Df, control=Cc, sigmas=sig, n=(nA, nS, nD, nC)),
           f"{R}/summary_cross_env.pt")
