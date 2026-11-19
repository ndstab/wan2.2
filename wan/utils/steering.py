"""T5.4 -- training-free weather steering of WanModel hidden states.

The paired atlas (paired_atlas.py) stores, for every (clear, weather) pair and
every (block, sigma) cell, the token-mean residual difference

    dmean[b, s] = mean_tokens( h_weather[b] - h_clear[b] )      # [dim]

Averaging dmean over many pairs gives a weather direction per (block, sigma).
Steering adds alpha * direction to the post-block residual of selected blocks,
only while the current sigma lies inside a chosen window, during an ordinary
generation. No gradients, no optimiser, no weight changes: the direction is a
mean of activations, and the hooks are registered from outside the model, so
WanModel.forward is untouched and the hooks are a no-op once detached.

alpha is in the atlas's own units: alpha = 1 adds exactly the average measured
clear->weather difference at that (block, sigma).
"""
import glob
import math
import os
import re

import torch

PARAM_KEYS = ('cloud', 'rain', 'fog', 'puddle', 'snow', 'snowflake')


# ---------------------------------------------------------------- records ----

def load_records(records_dir, drop_tokmaps=True):
    """Load paired-atlas per-pair records; skip content-control pairs."""
    recs = []
    for p in sorted(glob.glob(os.path.join(records_dir, '*.pt'))):
        r = torch.load(p, map_location='cpu', weights_only=False)
        if not isinstance(r, dict) or 'blocks' not in r:
            continue
        if str(r['variant']).startswith('vs'):      # --pair_mode content
            continue
        if drop_tokmaps:
            for cell in r['blocks'].values():
                cell.pop('tokmap', None)
        recs.append(r)
    if not recs:
        raise ValueError(f"no weather-pair records found in {records_dir}")
    return recs


_FILTER_RE = re.compile(r'^\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*([-\d.]+)\s*$')


def parse_filter(expr):
    """'fog>=5,rain==0' -> predicate on a params dict. Empty/None -> accept all."""
    if not expr:
        return lambda p: True
    ops = {'>=': float.__ge__, '<=': float.__le__, '==': float.__eq__,
           '!=': float.__ne__, '>': float.__gt__, '<': float.__lt__}
    terms = []
    for part in expr.split(','):
        m = _FILTER_RE.match(part)
        if not m or m.group(1) not in PARAM_KEYS:
            raise ValueError(f"bad filter term {part!r}; keys are {PARAM_KEYS}")
        terms.append((m.group(1), ops[m.group(2)], float(m.group(3))))
    return lambda p: all(op(float(p.get(k, 0.0)), v) for k, op, v in terms)


def _param_vec(params, scale):
    return torch.tensor([float(params.get(k, 0.0)) / scale[k] for k in PARAM_KEYS])


def param_scale(records):
    """Per-key max over records, so the kNN distance weights every factor alike."""
    return {k: max([float(r['params'].get(k, 0.0)) for r in records] + [1e-6])
            for k in PARAM_KEYS}


def nearest_records(records, target_params, k, exclude_shards=()):
    """The k records whose weather vector is closest to target_params."""
    pool = [r for r in records if r['shard'] not in set(exclude_shards)]
    scale = param_scale(records)
    tgt = _param_vec(target_params, scale)
    pool.sort(key=lambda r: (_param_vec(r['params'], scale) - tgt).norm().item())
    return pool[:k]


def build_directions(records, filt=None, exclude_shards=()):
    """Mean dmean over the selected records -> dict usable by WeatherSteerer."""
    pred = parse_filter(filt) if isinstance(filt, (str, type(None))) else filt
    sel = [r for r in records
           if r['shard'] not in set(exclude_shards) and pred(r['params'])]
    if not sel:
        raise ValueError("filter / exclusion left no records")
    sigmas = list(sel[0]['sigmas'])
    for r in sel:
        if any(abs(a - b) > 1e-6 for a, b in zip(r['sigmas'], sigmas)):
            raise ValueError(f"{r['shard']}_{r['variant']}: sigma grid differs")
    n_blocks = 1 + max(b for b, _ in sel[0]['blocks'])
    dim = sel[0]['blocks'][(0, 0)]['dmean'].numel()
    acc = torch.zeros(n_blocks, len(sigmas), dim, dtype=torch.float64)
    for r in sel:
        for (b, s), cell in r['blocks'].items():
            acc[b, s] += cell['dmean'].double()
    return dict(
        dirs=(acc / len(sel)).float(),           # [n_blocks, n_sigmas, dim]
        sigmas=sigmas,                           # descending, as in the atlas
        n_pairs=len(sel),
        pairs=[f"{r['shard']}_{r['variant']}" for r in sel],
        filter=filt if isinstance(filt, str) else None,
        exclude_shards=list(exclude_shards),
    )


# ---------------------------------------------------------------- steerer ----

class WeatherSteerer:
    """Adds alpha * direction(block, sigma) to post-block residuals.

    passes='both' steers every forward call (both CFG branches, matching how
    the reference K/V injection is applied). passes='cond' steers only the
    first call at each new sigma, which in WanTI2V.t2v() is the conditional
    branch (cond is called before uncond, textimage2video.py:515-519); the
    guide scale then amplifies the steering.
    """

    def __init__(self, directions, alpha=1.0, blocks=(19, 20, 21, 22),
                 sigma_window=(0.83, 0.95), norm_preserve=False, passes='both'):
        if passes not in ('both', 'cond'):
            raise ValueError(passes)
        self.dirs = directions['dirs']
        self.sigmas = [float(s) for s in directions['sigmas']]
        self.alpha = float(alpha)
        self.blocks = [int(b) for b in blocks]
        if any(b < 0 or b >= self.dirs.shape[0] for b in self.blocks):
            raise ValueError(f"blocks {self.blocks} outside [0, {self.dirs.shape[0]})")
        self.lo, self.hi = sorted(float(s) for s in sigma_window)
        self.norm_preserve = norm_preserve
        self.passes = passes
        self._handles = []
        self._sigma = None
        self._call_at_sigma = 0
        self._cache = {}
        self.n_applied = 0          # (block, call) pairs actually steered

    @classmethod
    def from_file(cls, path, **kw):
        return cls(torch.load(path, map_location='cpu', weights_only=False), **kw)

    def direction(self, block, sigma):
        """Linear interpolation in sigma between the atlas's measured levels."""
        key = (block, round(sigma, 6))
        if key in self._cache:
            return self._cache[key]
        s = self.sigmas
        order = sorted(range(len(s)), key=lambda i: s[i])      # ascending
        xs = [s[i] for i in order]
        if sigma <= xs[0]:
            v = self.dirs[block, order[0]]
        elif sigma >= xs[-1]:
            v = self.dirs[block, order[-1]]
        else:
            j = next(j for j in range(1, len(xs)) if xs[j] >= sigma)
            w = (sigma - xs[j - 1]) / (xs[j] - xs[j - 1])
            v = (1 - w) * self.dirs[block, order[j - 1]] + w * self.dirs[block, order[j]]
        self._cache[key] = v
        return v

    def active(self):
        if self._sigma is None or not (self.lo <= self._sigma <= self.hi):
            return False
        return self.passes == 'both' or self._call_at_sigma == 1

    def _pre_hook(self, _module, args, kwargs):
        t = kwargs['t'] if 't' in kwargs else args[1]
        sigma = float(t.detach().float().max()) / 1000.0
        if self._sigma is None or not math.isclose(sigma, self._sigma, abs_tol=1e-7):
            self._sigma = sigma
            self._call_at_sigma = 0
        self._call_at_sigma += 1

    def _make_block_hook(self, block):
        def hook(_module, _inp, out):
            if not self.active():
                return None
            v = self.direction(block, self._sigma).to(device=out.device,
                                                      dtype=out.dtype)
            steered = out + self.alpha * v
            if self.norm_preserve:
                n0 = out.float().norm(dim=-1, keepdim=True)
                n1 = steered.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                steered = (steered.float() * (n0 / n1)).to(out.dtype)
            self.n_applied += 1
            return steered
        return hook

    def attach(self, model):
        self.detach()
        self._handles.append(
            model.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for b in self.blocks:
            self._handles.append(
                model.blocks[b].register_forward_hook(self._make_block_hook(b)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._sigma = None
        self._call_at_sigma = 0
