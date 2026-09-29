"""Paper 2 §5-7: target-bridge ε-strong simulation (Algorithm 7.1) for additive-noise (constant covariance) models.

Likelihood oracle (Prop 6.6): with G(t,x) = -b(t,x)^T S (x-v)/(T-t)  (Delta a = 0 for constant sigma) and the
certified enclosure of the retained guided bridge,
    J(X°) in  Ipre + [-R_tail, R_tail],
    Ipre = sum over cells of (t_{i+1}-t_i) * G-enclosure on the cell box   (interval Riemann sum on [0,T-h]),
    R_tail = supb ‖S‖ M h^q / q                                          (eq. (45), rigorous in ideal arithmetic).
The preterminal box on cell i is ball(centre value, e_pre + cell oscillation) with e_pre = driver + solver error.
`oracle.new_proposal(rng)` returns a retained latent proposal whose `interval(n)` intersects successive enclosures.

Contracts kept: proposals are independent, all refinement extends the SAME latent guided bridge; the accepted
record (`TargetBridge`) keeps the proposal and can be refined further without re-running acceptance.
"""
from __future__ import annotations

import math

import numpy as np

from ..bridges.additive import AdditiveGuidedBridge, AdditiveModel, Enclosure
from ..rng import NULL_BUDGET, Budget, BudgetExhausted
from .abstract import LogEnvelope, Selection, retrospective_select
from . import moment as mom


class GuidedProposal:
    """Retained latent guided bridge + nested likelihood intervals."""

    LEVEL0 = 5
    MAX_LEVEL = 19          # explicit refinement-depth budget (memory ~2^level); exceeding raises BudgetExhausted

    def __init__(self, model: AdditiveModel, u, v, T, rng, budget: Budget = NULL_BUDGET, **kw):
        self.model, self.u, self.v, self.T = model, np.asarray(u, float), np.asarray(v, float), float(T)
        self.bridge = AdditiveGuidedBridge(model, u, v, T, rng, budget=budget, **kw)
        self.lo, self.hi = -math.inf, math.inf
        self.level_used = 0
        self.budget = budget

    def level_for(self, n: int) -> int:
        return self.LEVEL0 + n - 1

    def J_interval_at_level(self, lev: int) -> tuple[float, float]:
        br, m, T = self.bridge, self.model, self.T
        enc = br.enclosure(lev)
        q = lev + br.cut_extra
        h = T * 2.0 ** (-q)
        t, X = enc.times, enc.values
        # preterminal cells: those with t_{i+1} <= T-h  (last stored node before T is T-h; last entry is v at T)
        ts, Xs = t[:-1], X[:-1]
        e_pre = enc.parts["pre"]
        t0, t1 = ts[:-1], ts[1:]
        xc = 0.5 * (Xs[:-1] + Xs[1:])
        r = e_pre + 0.5 * np.linalg.norm(Xs[1:] - Xs[:-1], axis=1)
        lo, hi = _G_enclosure_vec(m, t0, t1, xc, r, self.v, T)
        dt = t1 - t0
        Ilo, Ihi = float(np.sum(dt * lo)), float(np.sum(dt * hi))
        Rt = br.tail_integral_bound(lev, q)
        return Ilo - Rt, Ihi + Rt

    def interval(self, n: int) -> tuple[float, float]:
        lev = self.level_for(n)
        if lev > self.MAX_LEVEL:
            raise BudgetExhausted(f"refinement level {lev} exceeds MAX_LEVEL={self.MAX_LEVEL}")
        if lev > self.level_used:
            lo, hi = self.J_interval_at_level(lev)
            self.lo, self.hi = max(self.lo, lo), min(self.hi, hi)
            self.level_used = lev
        return self.lo, self.hi


def _G_enclosure_vec(m: AdditiveModel, t0, t1, xc, r, v, T):
    """Vectorised G enclosure (b affine-Lipschitz):  see AdditiveModel.G_enclosure."""
    ou = m._ou
    bc = xc @ ou.B.T + ou.c
    y = xc - v
    core = -np.einsum("ij,jk,ik->i", bc, m.S, y)
    dev = (m.L * m.S_norm * np.linalg.norm(y, axis=1) * r + np.linalg.norm(bc, axis=1) * m.S_norm * r
           + m.L * m.S_norm * r * r)
    lo, hi = core - dev, core + dev
    tlo, thi = T - t1, T - t0
    c = np.stack([lo / tlo, lo / thi, hi / tlo, hi / thi])
    return c.min(0), c.max(0)


class OUGuidedOracle:
    """LogWeightOracle for the OU target bridge (guide = covariance-matched Brownian auxiliary)."""

    def __init__(self, ou, u, v, T, budget: Budget = NULL_BUDGET, **kw):
        self.ou, self.u, self.v, self.T = ou, np.asarray(u, float), np.asarray(v, float), float(T)
        self.model = AdditiveModel.from_ou(ou)
        self.budget, self.kw = budget, kw

    def new_proposal(self, rng):
        return GuidedProposal(self.model, self.u, self.v, self.T, rng, budget=self.budget, **self.kw)

    def layer_certificate(self, mode: str = "tight") -> dict:
        """mode='thm84': Paper 2 Theorem 8.4 (heat-kernel) constant; 'tight': model-specific Gaussian-moment constant
        (deterministic, from the model; float quadrature).  Both are valid inputs M_c to Algorithm 3.6."""
        if mode == "thm84":
            hk = mom.ou_heat_kernel_constants(self.ou, self.T)
            th = mom.ou_endpoint_data(self.ou, self.T, self.u, self.v)["theta_uT"]
            return mom.layer_certificate(self.ou.d, self.T, self.u, self.v, self.ou.a, hk, th)
        return mom.ou_tight_certificate(self.ou, self.T, self.u, self.v)


class TargetBridge:
    """Accepted record: exact retrospective selection done; refinement of the SAME latent path afterwards."""

    def __init__(self, sel: Selection):
        self.selection = sel
        self.proposal: GuidedProposal = sel.proposal          # type: ignore[assignment]

    @property
    def bridge(self) -> AdditiveGuidedBridge:
        return self.proposal.bridge

    def refine_to(self, eps: float, **kw) -> Enclosure:
        return self.bridge.refine_to(eps, **kw)

    def enclosure(self, n: int) -> Enclosure:
        return self.bridge.enclosure(n)

    def max_norm_interval(self, n: int) -> tuple[float, float]:
        return self.bridge.max_norm_interval(n)


def sample_target_bridge(oracle: OUGuidedOracle, rng, Mc: float | None = None, budget: Budget = NULL_BUDGET,
                         max_factory_coins: int | None = None) -> TargetBridge:
    """Algorithm 7.1 (steps 1-2): exact selection.  Refinement to a tolerance is then `TargetBridge.refine_to`."""
    if Mc is None:
        Mc = oracle.layer_certificate()["M_c"]
    sel = retrospective_select(oracle, LogEnvelope(Mc), rng, budget=budget, max_factory_coins=max_factory_coins)
    return TargetBridge(sel)
