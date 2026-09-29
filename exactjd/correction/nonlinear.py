"""Nonlinear benchmark: guided proposal (Paper 1, rough route) -> retrospective correction (Paper 2, Alg 7.1).

The target is the bridge of the papers' 2-d nonconstant-covariance, noncommuting diffusion; the proposal is the
covariance-matched guided bridge solved through the retained enhanced Brownian driver (`RoughGuidedBridge`), the weight
is Psi=exp(J), J=int_0^T G(t,X_t)dt with the FULL formula (36) (covariance mismatch Delta a != 0):

    G(t,x) = -b(x)^T S y/h - tr(Delta a(x) S)/(2h) + y^T S Delta a(x) S y/(2h^2),   y=x-v, h=T-t.

STATUS: **diagnostic / approximate**.  The certified Paper 1 pipeline for this model is unattainable (see experiments:
certified driver ball delta_n>1 up to n~90, block certificate limited to T~1e-5), so the likelihood interval below is
built from the level-L centre path with a calibrated, monotone, UNCERTIFIED half-width  w_L = cw 2^{-L/2}(1+|J_L|).
Every proposal/selection carries `status='diagnostic-UNCERTIFIED'`.  It exists to exercise the chain
guide -> correction -> jump skeleton and to compare the corrected law with the direct transformed reference.
"""
from __future__ import annotations

import math

import numpy as np

from ..bridges.rough_guided import RoughGuidedBridge
from ..bridges.roughmodels import NonlinearRoughModel
from ..rng import NULL_BUDGET, Budget


def G_values(model, ts: np.ndarray, xs: np.ndarray, v: np.ndarray, T: float) -> np.ndarray:
    """G(t,x) of eq. (36) on nodes (uses the Ito drift b = F + 1/2 sum D sigma_j sigma_j; nonlinear benchmark: b=(-sin(g(x1))/4,0))."""
    from ..models import nonlinear as nl
    out = np.empty(len(ts))
    S, aT = model.S, model.aT
    for i, (t, x) in enumerate(zip(ts, xs)):
        h = T - t
        y = x - v
        b = np.array([-0.25 * math.sin(nl.g_scalar(float(x[0]))), 0.0])
        da = model.a(x) - aT
        out[i] = (-(b @ S @ y) / h - 0.5 * np.trace(da @ S) / h + 0.5 * (y @ S @ da @ S @ y) / h**2)
    return out


class NonlinearGuidedProposal:
    status = "diagnostic-UNCERTIFIED"
    LEVEL0 = 5
    MAX_LEVEL = 11
    CW = 2.0

    def __init__(self, u, v, T, rng, budget: Budget = NULL_BUDGET):
        self.model = NonlinearRoughModel()
        self.bridge = RoughGuidedBridge(self.model, u, v, T, rng, budget=budget)
        self.u, self.v, self.T = np.asarray(u, float), np.asarray(v, float), float(T)
        self.lo, self.hi = -math.inf, math.inf
        self._lev = 0

    def J_center(self, L: int) -> float:
        q = max(2, L - 3)
        ts, xs = self.bridge.center(L, q, 2)
        G = G_values(self.bridge.model, ts, xs, self.v, self.T)
        J = float(np.sum(0.5 * (G[1:] + G[:-1]) * np.diff(ts)))
        # terminal part: G ~ tau^{-1/2}; estimate int_{T-h}^{T}|G| ~ 2 |G|_last sqrt(tau_last) sqrt(h)
        h = self.T - ts[-1]
        tail = 2.0 * abs(G[-1]) * math.sqrt(h) * math.sqrt(h)
        return J + 0.0 * tail, tail

    def interval(self, n: int):
        from ..rng import BudgetExhausted
        L = self.LEVEL0 + n - 1
        if L > self.MAX_LEVEL:
            raise BudgetExhausted(f"nonlinear proposal level {L} beyond MAX_LEVEL={self.MAX_LEVEL}")
        if L > self._lev:
            J, tail = self.J_center(L)
            w = self.CW * 2.0 ** (-L / 2) * (1 + abs(J)) + tail
            self.lo, self.hi = max(self.lo, J - w), min(self.hi, J + w)
            self._lev = L
        return self.lo, self.hi


class NonlinearGuidedOracle:
    def __init__(self, u, v, T, budget: Budget = NULL_BUDGET):
        self.u, self.v, self.T, self.budget = np.asarray(u, float), np.asarray(v, float), float(T), budget

    def new_proposal(self, rng):
        return NonlinearGuidedProposal(self.u, self.v, self.T, rng, self.budget)
