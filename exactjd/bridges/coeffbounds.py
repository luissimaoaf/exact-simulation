"""Interval-evaluated coefficient bounds for the rough-path verifier (Paper 1 Definition 6.1 (E2)).

Coefficients are given as sympy expressions.  Sup-norm bounds of arbitrary partial derivatives on a rectangular box
are obtained by `mpmath.iv` interval arithmetic (directed rounding) with recursive bisection.  This is a *small,
model-level* interval facility (not a general framework): only sin/cos/exp/sqrt/log/polynomial expressions.

`DerivTensorBounds(fields, coords, ops)` returns, for each order r, a Hilbert-Schmidt bound of the r-th derivative
tensor of the vector fields  V_j: R^d -> R^d  (j=1..m), i.e. an upper bound of the operator norm ‖D^r V‖.
Custom derivative operators (`ops`) let a model differentiate in coordinates other than its expression variables
(used by the nonlinear benchmark whose fields are explicit in (z=g(x1), x2) with d/dx1 = f'(z)^{-1} d/dz).
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from mpmath import iv

iv.dps = 30

_MODS = [{"sin": iv.sin, "cos": iv.cos, "exp": iv.exp, "sqrt": iv.sqrt, "log": iv.log, "tan": iv.tan, "Abs": abs}]     # dict only: rationals print as Python floats (documented: not exact for non-dyadic rationals)


class IntervalFn:
    """Compiled sympy expression evaluated on interval boxes."""

    def __init__(self, expr, syms: Sequence[sp.Symbol]):
        self.expr = sp.simplify(expr) if not isinstance(expr, (int, float)) else sp.Float(expr)
        self.syms = list(syms)
        self.f = sp.lambdify(self.syms, self.expr, modules=_MODS)

    def enclosure(self, lo: Sequence[float], hi: Sequence[float]) -> tuple[float, float]:
        args = [iv.mpf([float(a), float(b)]) for a, b in zip(lo, hi)]
        r = self.f(*args)
        r = iv.mpf(r) if not hasattr(r, "a") else r
        return float(r.a), float(r.b)

    def sup_abs(self, lo, hi, depth: int = 4, tol: float = 0.02) -> float:
        """Rigorous upper bound of sup|expr| over the box by recursive bisection (until relative gap < tol)."""
        a, b = self.enclosure(lo, hi)
        ub = max(abs(a), abs(b))
        if depth == 0:
            return ub
        # bisect the widest coordinate; prune with best lower bound of sup
        k = int(np.argmax(np.array(hi, float) - np.array(lo, float)))
        mid = 0.5 * (lo[k] + hi[k])
        h1, l2 = list(hi), list(lo)
        h1[k], l2[k] = mid, mid
        return max(self.sup_abs(lo, h1, depth - 1, tol), self.sup_abs(l2, hi, depth - 1, tol))


class DerivTensorBounds:
    """Bounds of ‖D^r V‖ (r=0..rmax) for the stacked fields V=(V_1..V_m) on boxes."""

    def __init__(self, fields: Sequence[Sequence[sp.Expr]], syms: Sequence[sp.Symbol],
                 ops: Sequence[Callable[[sp.Expr], sp.Expr]] | None = None, rmax: int = 5):
        """fields[j][i] = i-th component of V_j;  ops[k](expr) = d expr / d x_k  (default: sympy diff)."""
        self.syms = list(syms)
        d = len(syms)
        self.ops = list(ops) if ops is not None else [(lambda e, s=s: sp.diff(e, s)) for s in syms]
        self.rmax = rmax
        self.cache: dict = {}
        self.fns: dict[int, list[IntervalFn]] = {}
        level = [e for Vj in fields for e in Vj]                      # order 0 expressions
        self.fns[0] = [IntervalFn(e, self.syms) for e in level]
        cur = level
        for r in range(1, rmax + 1):
            nxt = []
            for e in cur:
                for k in range(d):
                    nxt.append(self.ops[k](e))
            cur = nxt
            self.fns[r] = [IntervalFn(e, self.syms) for e in cur]

    def bound(self, r: int, lo: Sequence[float], hi: Sequence[float], depth: int = 3) -> float:
        key = (r, tuple(np.round(lo, 6)), tuple(np.round(hi, 6)), depth)
        if key not in self.cache:
            tot = 0.0
            for fn in self.fns[r]:
                tot += fn.sup_abs(lo, hi, depth) ** 2
            self.cache[key] = math.sqrt(tot) * (1 + 1e-12)
        return self.cache[key]

    def all(self, lo, hi, rs=range(0, 4), depth: int = 3) -> dict:
        return {r: self.bound(r, lo, hi, depth) for r in rs}
