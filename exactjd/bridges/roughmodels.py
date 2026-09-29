"""Rough-path models: fields V_j (columns of sigma), Stratonovich drift F, covariance-matched guide, interval bounds.

Guided Stratonovich equation (Paper 1 (7)):  dX = [ -A(X)(X-v)/(T-t) + F(X) ] dt + sum_j V_j(X) dU^j,
A(x)=a(x) a_T^{-1},  a=sigma sigma^T,  F=b-(1/2) sum_j D sigma_j sigma_j  (Ito->Stratonovich correction).

`SymbolicRoughModel` compiles sympy expressions (in model variables) to numpy for evaluation and to interval
functions for derivative bounds.  Two models:
  * NonlinearRoughModel : the papers' 2-d nonconstant-covariance, noncommuting benchmark in x-coordinates
    (variables (z=g(x1), x2); d/dx1 = f'(z)^{-1} d/dz).  Fields are 2pi-periodic in (z,x2): bounds are GLOBAL.
  * HestonRoughModel    : log-price/variance Heston (Paper 1 §9.3) on D=R x (0,inf); bounds are LOCAL on boxes
    with z>=l>0 (fifth derivative of sqrt(z) grows like z^{-9/2}).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from ..models import nonlinear as nl
from .coeffbounds import DerivTensorBounds, IntervalFn
from .cpverify import Bounds


class SymbolicRoughModel:
    d = 2
    m = 2

    def __init__(self, syms, V_exprs, F_exprs, a_exprs, ops=None, to_vars=None, box_vars=None):
        """V_exprs[j] = [V_j^1..V_j^d]; F_exprs = [F^1..F^d]; a_exprs = d x d list; syms: model variables."""
        self.syms = list(syms)
        self.ops = ops
        self.to_vars = to_vars or (lambda x: x)
        self.box_vars = box_vars or (lambda lo, hi: (lo, hi))
        self._V = sp.lambdify(self.syms, sp.Matrix(V_exprs).T, "numpy")     # (d,m)
        self._F = sp.lambdify(self.syms, sp.Matrix(F_exprs), "numpy")
        self._a = sp.lambdify(self.syms, sp.Matrix(a_exprs), "numpy")
        self.Vb = DerivTensorBounds(V_exprs, self.syms, ops, rmax=5)
        self.Fb = DerivTensorBounds([F_exprs], self.syms, ops, rmax=1)
        self._V_exprs, self._F_exprs, self._a_exprs = V_exprs, F_exprs, a_exprs
        self.v = None
        self.T = None

    # ---- numeric evaluation
    def V(self, x):
        return np.array(self._V(*self.to_vars(np.asarray(x, float))), float)

    def F(self, x):
        return np.array(self._F(*self.to_vars(np.asarray(x, float))), float).reshape(-1)

    def a(self, x):
        return np.array(self._a(*self.to_vars(np.asarray(x, float))), float)

    def bind(self, v, T):
        self.v, self.T = np.asarray(v, float), float(T)
        self.aT = self.a(self.v)
        self.S = np.linalg.inv(self.aT)
        self._A_exprs = None
        return self

    def A(self, x):
        return self.a(x) @ self.S

    def f(self, t, x):
        return -self.A(x) @ (x - self.v) / (self.T - t) + self.F(x)

    def drift_b(self, x):
        """Ito drift b = F + 1/2 sum_j D sigma_j sigma_j (numerical differentiation of V, small step)."""
        x = np.asarray(x, float)
        eps = 1e-6
        Vx = self.V(x)
        corr = np.zeros(self.d)
        for j in range(self.m):
            J = np.stack([(self.V(x + eps * e)[:, j] - self.V(x - eps * e)[:, j]) / (2 * eps) for e in np.eye(self.d)], axis=1)
            corr += J @ Vx[:, j]
        return self.F(x) + 0.5 * corr

    # ---- interval bounds on x-boxes  (|x-c|_inf <= r)
    def _box(self, c, r):
        lo, hi = np.asarray(c, float) - r, np.asarray(c, float) + r
        return self.box_vars(lo, hi)

    def V_bounds(self, c, r, rs=range(0, 6), depth: int = 2):
        lo, hi = self._box(c, r)
        return {k: self.Vb.bound(k, lo, hi, depth) for k in rs}

    def coeff_bounds(self, t0: float, t1: float, c, r: float) -> Bounds:
        """Bounds for the Picard test on the time block [t0,t1], box |x-c|_inf<=r (autonomous fields: theta=0)."""
        Kv = self.V_bounds(c, r, range(0, 4))
        lo, hi = self._box(c, r)
        Fn0, Fn1 = self.Fb.bound(0, lo, hi, 2), self.Fb.bound(1, lo, hi, 2)
        # ||A|| and ||DA|| via the rows of A treated numerically on the box corners is not rigorous; use interval bounds
        An0, An1 = self._A_bounds(lo, hi)
        y = np.linalg.norm(np.asarray(c, float) - self.v) + math.sqrt(self.d) * r
        tau = self.T - t1
        F0 = An0 * y / tau + Fn0
        F1 = (An1 * y + An0) / tau + Fn1
        return Bounds((Kv[0], Kv[1], Kv[2], Kv[3]), (0.0, 0.0, 0.0), F0, F1)

    def _A_bounds(self, lo, hi):
        if self._A_exprs is None:
            A = sp.Matrix(self._a_exprs) * sp.Matrix(self.S.tolist())
            cols = [[A[i, j] for i in range(self.d)] for j in range(self.d)]
            self._A_exprs = DerivTensorBounds(cols, self.syms, self.ops, rmax=1)
        return self._A_exprs.bound(0, lo, hi, 2), self._A_exprs.bound(1, lo, hi, 2)


# ---------------------------------------------------------------------------------------------------- nonlinear
class _NonlinearFast(SymbolicRoughModel):
    """Same model; numeric V, F, a use scalar closed forms (sympy is kept for the interval bounds only)."""

    def V(self, x):
        q = 1.0 + 0.5 * math.cos(nl.g_scalar(float(x[0])))
        c, s = math.cos(x[1]), math.sin(x[1])
        return np.array([[q * c, -q * s], [s, c]])

    def F(self, x):
        return np.array([0.5 * (1.0 + 0.5 * math.cos(nl.g_scalar(float(x[0])))), 0.0])

    def a(self, x):
        q = 1.0 + 0.5 * math.cos(nl.g_scalar(float(x[0])))
        return np.array([[q * q, 0.0], [0.0, 1.0]])


_NL_PROTO: list = []
_HESTON_PROTO: dict = {}


def NonlinearRoughModel() -> SymbolicRoughModel:
    """Cached symbolic prototype (building the sympy derivative tree takes ~10 s); callers get a shallow copy so that
    `bind(v,T)` state is private while the compiled interval functions and their caches are shared."""
    import copy
    if not _NL_PROTO:
        _NL_PROTO.append(_build_nonlinear())
    return copy.copy(_NL_PROTO[0])


def _build_nonlinear() -> SymbolicRoughModel:
    z, x2 = sp.symbols("z x2", real=True)
    fp = 1 + sp.cos(z) / 2
    q = fp                                    # q(x1)=f'(g(x1)) = 1 + cos(z)/2  with z=g(x1)
    V1 = [q * sp.cos(x2), sp.sin(x2)]
    V2 = [-q * sp.sin(x2), sp.cos(x2)]
    F = [q / 2, sp.Integer(0)]
    a = [[q**2, 0], [0, 1]]
    ops = [lambda e: sp.diff(e, z) / fp, lambda e: sp.diff(e, x2)]
    return _NonlinearFast([z, x2], [V1, V2], F, a, ops,
                              to_vars=lambda x: (float(nl.g(x[0])), float(x[1])),
                              box_vars=lambda lo, hi: ([float(nl.g(lo[0])), lo[1]], [float(nl.g(hi[0])), hi[1]]))


# ---------------------------------------------------------------------------------------------------- Heston
def HestonRoughModel(kappa: float, theta: float, xi: float, rho: float, r: float = 0.0) -> SymbolicRoughModel:
    import copy
    key = (kappa, theta, xi, rho, r)
    if key not in _HESTON_PROTO:
        _HESTON_PROTO[key] = _build_heston(kappa, theta, xi, rho, r)
    return copy.copy(_HESTON_PROTO[key])


def _build_heston(kappa, theta, xi, rho, r) -> SymbolicRoughModel:
    p = sp.Symbol("p", real=True)
    zz = sp.Symbol("z", positive=True)
    c = math.sqrt(1 - rho**2)
    s = sp.sqrt(zz)
    V1 = [s, xi * rho * s]
    V2 = [sp.Integer(0), xi * c * s]
    F = [r - zz / 2 - xi * rho / 4, kappa * (theta - zz) - xi**2 / 4]
    a = [[zz, xi * rho * zz], [xi * rho * zz, xi**2 * zz]]
    return SymbolicRoughModel([p, zz], [V1, V2], F, a)
