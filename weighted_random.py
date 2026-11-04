"""Weighted random hyperparameter search.

Generalised version of the search loop in ``notebooks/Weighted_Random_CIFAR.ipynb``.

How it works
------------
1. Two starting configurations are placed at roughly 1/3 and 2/3 of every
   parameter's range (measured in log space for log-scaled parameters) and
   both are evaluated.
2. Each iteration proposes one new configuration.  For every parameter the two
   best values seen so far form a bracket, which is widened on both sides so
   the search can leave the bracket instead of locking into a local optimum
   the way a fixed grid does.  The bracket is split at its midpoint and each
   half is cut into a number of candidate points proportional to the *other*
   half's loss, so values near the better of the two bests are sampled more
   often.  One candidate is drawn uniformly from the pooled points.
3. The new configuration is evaluated and, if it beats either of the two
   bests, replaces it.  Repeat.

The objective is any callable ``config -> loss`` (or ``config -> (loss, extra)``),
so the same search drives the Keras CNN from the notebook and the DARTS
search in ``tune.py``.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


def _divisions(lo: float, hi: float, n: int) -> List[float]:
    """Midpoints of ``n`` equal-width bins spanning [lo, hi]."""
    width = (hi - lo) / n
    return [lo + width * (i + 0.5) for i in range(n)]


@dataclass
class Param:
    """A single search dimension.

    log      sample in log space (base 3, as in the original notebook)
    integer  round proposals to the nearest integer
    bounds   optional hard (lo, hi) clamp applied after the bracket is widened;
             use ``None`` for an open side, e.g. ``(0, None)``
    """

    low: float
    high: float
    log: bool = False
    integer: bool = False
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None

    def _fwd(self, v: float) -> float:
        return math.log(v, 3) if self.log else v

    def _inv(self, v: float) -> float:
        return 3.0 ** v if self.log else v

    def clip(self, v: float):
        lo, hi = self.bounds if self.bounds else (None, None)
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return int(round(v)) if self.integer else v

    def init_pair(self):
        """The two starting values at 1/3 and 2/3 of the range."""
        a, b = self._fwd(self.low), self._fwd(self.high)
        return self.clip(self._inv(a + 0.33 * (b - a))), self.clip(self._inv(a + 0.67 * (b - a)))

    def propose(self, v1, loss1, v2, loss2, rng: random.Random, widen=0.5, n_points=100, min_pad=0.05):
        """Draw a new value from the loss-weighted, widened bracket [v1, v2]."""
        if v1 <= v2:
            lo, lo_loss, hi, hi_loss = v1, loss1, v2, loss2
        else:
            lo, lo_loss, hi, hi_loss = v2, loss2, v1, loss1

        # Points on the low half are proportional to the HIGH value's loss and
        # vice versa: the worse one side is, the denser we sample the other.
        total = lo_loss + hi_loss
        n_lo = int(round(n_points * hi_loss / total)) if total > 0 else n_points // 2
        n_lo = min(max(n_lo, 1), n_points - 1)
        n_hi = n_points - n_lo

        lo, hi = self._fwd(lo), self._fwd(hi)
        mid = (lo + hi) / 2
        # widen the bracket past both bests; never below a small floor so the
        # search keeps exploring once the two bests have converged
        pad = max(widen * (hi - lo), min_pad * abs(mid), 1e-3)
        lo, hi = lo - pad, hi + pad

        candidates = _divisions(lo, mid, n_lo) + _divisions(mid, hi, n_hi)
        return self.clip(self._inv(rng.choice(candidates)))


class WeightedRandomSearch:
    """Run the weighted random search over a dict of :class:`Param`.

    objective(config) must return a loss to minimise, or ``(loss, extra)``
    where ``extra`` is a JSON-serialisable dict kept in the history.
    """

    def __init__(self, space: Dict[str, Param], objective: Callable, n_iter: int = 5,
                 seed: int = 0, widen: float = 0.5, n_points: int = 100,
                 log_path: Optional[str] = None, verbose: bool = True):
        self.space = space
        self.objective = objective
        self.n_iter = n_iter
        self.rng = random.Random(seed)
        self.widen = widen
        self.n_points = n_points
        self.log_path = log_path
        self.verbose = verbose
        self.history: List[dict] = []
        self.best: Optional[dict] = None
        self.second: Optional[dict] = None

    # ------------------------------------------------------------------ helpers
    def _evaluate(self, config: dict, tag: str) -> dict:
        out = self.objective(dict(config))
        loss, extra = (out if isinstance(out, tuple) else (out, {}))
        trial = {'trial': len(self.history), 'tag': tag, 'config': config, 'loss': float(loss), **extra}
        self.history.append(trial)
        if self.verbose:
            print(f"[wrs] trial {trial['trial']:>3} ({tag}) loss={loss:.5f} config={config}")
        self._dump()
        return trial

    def _dump(self):
        if self.log_path:
            with open(self.log_path, 'w') as f:
                json.dump({'history': self.history, 'best': self.best, 'second': self.second}, f, indent=2, default=str)

    def _update_best(self, trial: dict):
        if trial['loss'] < self.best['loss']:
            self.best, self.second = trial, self.best
        elif trial['loss'] < self.second['loss']:
            self.second = trial

    # --------------------------------------------------------------------- run
    def propose(self) -> dict:
        return {name: p.propose(self.best['config'][name], self.best['loss'],
                                self.second['config'][name], self.second['loss'],
                                self.rng, self.widen, self.n_points)
                for name, p in self.space.items()}

    def run(self) -> dict:
        pairs = {name: p.init_pair() for name, p in self.space.items()}
        t1 = self._evaluate({k: v[0] for k, v in pairs.items()}, 'init-1/3')
        t2 = self._evaluate({k: v[1] for k, v in pairs.items()}, 'init-2/3')
        self.best, self.second = (t1, t2) if t1['loss'] <= t2['loss'] else (t2, t1)
        self._dump()

        for i in range(self.n_iter):
            trial = self._evaluate(self.propose(), f'iter-{i + 1}')
            self._update_best(trial)
            self._dump()
            if self.verbose:
                print(f"[wrs]   best so far: loss={self.best['loss']:.5f} (trial {self.best['trial']})")
        return self.best


if __name__ == '__main__':
    # Tiny self-check on a toy objective with a known optimum: lr=10**-2.5, dropout=0.3.
    space = {
        'lr': Param(1e-4, 1e-1, log=True),
        'dropout': Param(0.01, 0.5, bounds=(0.0, 0.9)),
        'epochs': Param(5, 25, integer=True, bounds=(1, None)),
    }

    def toy(cfg):
        return (math.log10(cfg['lr']) + 2.5) ** 2 + (cfg['dropout'] - 0.3) ** 2 + 0.001 * cfg['epochs']

    best = WeightedRandomSearch(space, toy, n_iter=30, seed=0).run()
    print('best:', best)
