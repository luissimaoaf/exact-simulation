"""Bridge-record factories = contract (D3) of Paper 3, and a `JumpSkeleton` output container.

A *bridge record* is one retained latent path between exact endpoints with nested certified enclosures:
    rec.enclosure(level) -> `PathEnc` (times, values, radius) with ‖latent - centre‖∞ <= radius,
    rec.max_norm_interval(level) -> certified enclosure of sup_t |latent_t| (nested),
    rec.refine_to(eps)  -> enclosure with radius <= eps  (same latent path, never resampled).

Factories
  OUBridgeFactory          : isotropic OU, exact bridge = time-changed Brownian bridge (Paper 3 Prop 6.2, eq. (20)).
  TransformedBridgeFactory : nonlinear benchmark, bridge = Phi(Brownian bridge) (DIRECT reference route, not Papers 1-2).
  P2BridgeFactory          : bridges from Paper 2's exact retrospective selection of the guided proposal (chain route);
                             optional explicitly-labelled truncated race (`race_kmax`) for feasibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..brownian.record import BrownianRecord
from ..models import nonlinear as nl
from ..rng import NULL_BUDGET, Budget, BudgetExhausted


@dataclass
class PathEnc:
    times: np.ndarray
    values: np.ndarray
    radius: float
    level: int = 0

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return np.stack([np.interp(t, self.times, self.values[:, j]) for j in range(self.values.shape[1])], axis=-1)


class _BaseRecord:
    max_level = 22

    def refine_to(self, eps: float, n0: int = 3) -> PathEnc:
        n = n0
        while n <= self.max_level:
            e = self.enclosure(n)
            if e.radius <= eps:
                return e
            n += 1
        raise BudgetExhausted(f"tolerance {eps} not reached by level {self.max_level} (radius {e.radius:.3g})")



class _AdaptiveSup:
    """Adaptive certified sup-norm tracker on `self.rec` (a BrownianRecord bridge tree).

    Leaves are dyadic cells with EXACT latent endpoint values; the deviation of the latent path from a leaf's chord is
    at most rec.radius(depth) (tail-first per-level bounds).  Each round refines only leaves whose certified upper bound
    lies in the top half of the current interval [lo, hi], drawing single-cell noises lazily (consistent with later
    global refinement).  lo = max exact node norm, hi = max leaf upper bound; both nested.  Never resamples."""

    rounds_per_level = 2

    def _adaptive_init(self):
        rec = self.rec
        K = rec.Kstar
        nodes = rec.nodes(K)
        m = 2 ** K
        self._depth = np.full(m, K, dtype=int)
        self._idx = np.arange(m)
        self._w0, self._w1 = nodes[:-1].copy(), nodes[1:].copy()
        q = np.linspace(0.0, rec.h, m + 1)
        self._sup_lo = float(np.max(self._node_norms(nodes, q)))
        self._sup_hi = math.inf
        self._rounds = 0
        self._remc: dict = {}

    def _rem(self, depth: int) -> float:
        if depth not in self._remc:
            self._remc[depth] = self.rec.radius(depth)
        return self._remc[depth]

    def _sup_round(self):
        rec = self.rec
        L = rec.h * 2.0 ** (-self._depth.astype(float))
        q0 = self._idx * L
        rem = np.array([self._rem(int(dp)) for dp in self._depth])
        up = self._cell_upper(self._w0, self._w1, q0, L, rem)
        self._sup_hi = min(self._sup_hi, float(up.max()))
        sel = np.nonzero(up >= 0.5 * (self._sup_lo + up.max()))[0]
        keep = np.ones(len(up), bool)
        keep[sel] = False
        nd, ni, n0, n1 = [self._depth[keep]], [self._idx[keep]], [self._w0[keep]], [self._w1[keep]]
        mids, qm = [], []
        cd, ci, c0, c1 = [], [], [], []
        for j in sel:
            k = int(self._depth[j]) + 1
            xi = rec.xi_entry(k, int(self._idx[j]))
            mid = 0.5 * (self._w0[j] + self._w1[j]) + 0.5 * math.sqrt(L[j]) * xi
            mids.append(mid)
            qm.append(q0[j] + 0.5 * L[j])
            cd += [k, k]
            ci += [2 * int(self._idx[j]), 2 * int(self._idx[j]) + 1]
            c0 += [self._w0[j], mid]
            c1 += [mid, self._w1[j]]
        self._depth = np.concatenate(nd + [np.array(cd, dtype=int)])
        self._idx = np.concatenate(ni + [np.array(ci, dtype=int)])
        self._w0 = np.vstack(n0 + [np.array(c0)]) if c0 else self._w0[keep]
        self._w1 = np.vstack(n1 + [np.array(c1)]) if c1 else self._w1[keep]
        if mids:
            self._sup_lo = max(self._sup_lo, float(np.max(self._node_norms(np.array(mids), np.array(qm)))))
        self._rounds += 1

    def max_norm_interval(self, n: int) -> tuple[float, float]:
        if not hasattr(self, "_depth"):
            self._adaptive_init()
            self._sup_round()
        target = self.rounds_per_level * n
        while self._rounds < target:
            self._sup_round()
        return self._sup_lo, self._sup_hi

# --------------------------------------------------------------------------------------------- OU direct bridge
class OUBridgeRecord(_AdaptiveSup, _BaseRecord):
    """Y_{s+u} = e^{-kappa u}(x + Sigma Btilde_{q(u)}),  Btilde Brownian bridge on [0,q(h)] from 0 to v=Sigma^{-1}(e^{kappa h}y-x)."""

    def __init__(self, ou, s, x, t, y, rng):
        self.kappa = -float(ou.B[0, 0])
        assert np.allclose(ou.B, -self.kappa * np.eye(ou.d)) and ou.Sigma.shape[0] == ou.Sigma.shape[1]
        self.Sigma, self.s, self.t = ou.Sigma, float(s), float(t)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        h = self.t - self.s
        self.h = h
        k = self.kappa
        self.qh = h if k == 0 else (math.expm1(2 * k * h)) / (2 * k)
        v = np.linalg.solve(self.Sigma, math.exp(k * h) * self.y - self.x)
        self.rec = BrownianRecord(self.qh, ou.d, rng, a=np.zeros(ou.d), b=v)
        self.sig_norm = float(np.linalg.norm(self.Sigma, 2))
        self._cache: dict = {}

    def _u_of_q(self, q):
        k = self.kappa
        return q if k == 0 else np.log1p(2 * k * q) / (2 * k)

    def enclosure(self, n: int) -> PathEnc:
        if n in self._cache:
            return self._cache[n]
        nodes = self.rec.nodes(n)                                    # (2^n+1, d) values of B~ at q-grid
        qs = np.linspace(0.0, self.qh, len(nodes))
        us = self._u_of_q(qs)
        vals = np.exp(-self.kappa * us)[:, None] * (self.x[None, :] + nodes @ self.Sigma.T)
        vals[0], vals[-1] = self.x, self.y
        enc = PathEnc(self.s + us, vals, self.sig_norm * self.rec.radius(n), n)
        self._cache[n] = enc
        return enc

    # adaptive sup hooks (Y_u = e^{-kappa u}(x + Sigma B~_{q(u)}), e^{-kappa u} nonincreasing)
    def _node_norms(self, w, q):
        u = self._u_of_q(np.asarray(q, float))
        return np.linalg.norm(np.exp(-self.kappa * u)[:, None] * (self.x[None, :] + w @ self.Sigma.T), axis=1)

    def _cell_upper(self, w0, w1, q0, L, rem):
        z0 = np.linalg.norm(self.x[None, :] + w0 @ self.Sigma.T, axis=1)
        z1 = np.linalg.norm(self.x[None, :] + w1 @ self.Sigma.T, axis=1)
        return np.exp(-self.kappa * self._u_of_q(q0)) * (np.maximum(z0, z1) + self.sig_norm * rem)


class OUBridgeFactory:
    def __init__(self, ou, budget: Budget = NULL_BUDGET):
        self.ou, self.budget = ou, budget

    def new_bridge(self, s, x, t, y, rng):
        self.budget.tick("bridge_record")
        return OUBridgeRecord(self.ou, s, x, t, y, rng)


# --------------------------------------------------------------------------------------------- nonlinear (direct)
class TransformedBridgeRecord(_AdaptiveSup, _BaseRecord):
    """Phi(Brownian bridge from Psi(x) to Psi(y)); Phi is 3/2-Lipschitz coordinatewise (f'<=3/2)."""

    def __init__(self, s, x, t, y, rng):
        self.s, self.t = float(s), float(t)
        zx, zy = nl.Psi(np.asarray(x, float)), nl.Psi(np.asarray(y, float))
        self.rec = BrownianRecord(self.t - self.s, 2, rng, a=zx, b=zy)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self._cache: dict = {}

    def enclosure(self, n: int) -> PathEnc:
        if n in self._cache:
            return self._cache[n]
        nodes = self.rec.nodes(n)
        vals = nl.Phi(nodes)
        vals[0], vals[-1] = self.x, self.y
        # image polygon vs Phi(z-polygon): f between chord and graph deviates <= sup|f''| Delta^2/8 = Delta^2/16
        cur = float(np.max(np.abs(np.diff(nodes[:, 0])))) ** 2 / 16.0
        enc = PathEnc(np.linspace(self.s, self.t, len(nodes)), vals, 1.5 * self.rec.radius(n) + cur, n)
        self._cache[n] = enc
        return enc

    # adaptive sup hooks: Phi=(f(z1),z2) is 3/2-Lipschitz; |Phi(z(t))| <= |Phi(z_i)| + 1.5(|Delta|/2 + rem)
    def _node_norms(self, w, q):
        return np.linalg.norm(nl.Phi(w), axis=1)

    def _cell_upper(self, w0, w1, q0, L, rem):
        n0, n1 = np.linalg.norm(nl.Phi(w0), axis=1), np.linalg.norm(nl.Phi(w1), axis=1)
        dz = np.linalg.norm(w1 - w0, axis=1)
        return np.maximum(n0, n1) + 1.5 * (0.5 * dz + rem)


class TransformedBridgeFactory:
    def __init__(self, budget: Budget = NULL_BUDGET):
        self.budget = budget

    def new_bridge(self, s, x, t, y, rng):
        self.budget.tick("bridge_record")
        return TransformedBridgeRecord(s, x, t, y, rng)


# --------------------------------------------------------------------------------------------- Paper 2 chain
class P2BridgeRecord(_BaseRecord):
    """Record wrapping an accepted Paper-2 target bridge (exact retrospective selection done at construction).
    The path lives on the *normalised* time axis of the OU segment [s,t] itself (guide coefficients used as is)."""

    max_level = 19

    def __init__(self, tb, s, t):
        self.tb, self.s, self.t = tb, float(s), float(t)
        self._cache: dict = {}

    def enclosure(self, n: int) -> PathEnc:
        lev = max(n + 2, 5)
        if lev > self.max_level:
            raise BudgetExhausted(f"P2 record refinement level {lev} beyond {self.max_level}")
        if lev in self._cache:
            return self._cache[lev]
        e = self.tb.enclosure(lev)
        enc = PathEnc(self.s + e.times, e.values, e.radius, lev)
        self._cache[lev] = enc
        return enc

    def max_norm_interval(self, n: int) -> tuple[float, float]:
        return self.enclosure(n).sup_norm_interval() if hasattr(self.enclosure(n), "sup_norm_interval") else self._sup(n)

    def _sup(self, n: int) -> tuple[float, float]:
        e = self.enclosure(n)
        m = float(np.max(np.linalg.norm(e.values, axis=1)))
        return max(m - e.radius, 0.0), m + e.radius


class P2BridgeFactory:
    """Bridges by Paper 2's Algorithm 7.1 for an additive-noise OU model.

    mode='exact'     : paper's fixed log envelope (infinite-mean race; work limits raise BudgetExhausted).
    mode='truncated' : APPROXIMATE, explicitly labelled: bands k>race_kmax discarded; total-variation error of the
                       selected law is bounded by 2 P_P(J > race_kmax ln 2) <= 2 E_P[J^2]/(race_kmax ln2)^2 (Chebyshev).
    """

    def __init__(self, ou, mode: str = "exact", race_kmax: int = 3, race_kmin: int = -10,
                 max_factory_coins: int | None = 20000, budget: Budget = NULL_BUDGET,
                 certification: str = "certified"):
        self.ou, self.mode, self.kmax, self.kmin = ou, mode, race_kmax, race_kmin
        self.certification = certification
        self.max_factory_coins, self.budget = max_factory_coins, budget
        self.diagnostics: list = []
        self.tv_bounds: list = []

    def new_bridge(self, s, x, t, y, rng):
        from ..correction.bridge import OUGuidedOracle, TargetBridge
        from ..correction.abstract import LogEnvelope, retrospective_select
        from ..correction import moment as mom
        self.budget.tick("bridge_record")
        h = t - s
        orc = OUGuidedOracle(self.ou, np.asarray(x, float), np.asarray(y, float), h, budget=self.budget,
                             certification=self.certification, estimate_solver=(self.certification == "certified"))
        cert = mom.ou_tight_certificate(self.ou, h, x, y)
        if self.mode == "exact":
            env = LogEnvelope(cert["M_c"])
        else:
            env = _TruncatedEnvelope(cert["M_c"], self.kmin, self.kmax)
            xcut = min(self.kmax, -self.kmin - 1) * math.log(2)     # |J| > xcut  <=>  outside the retained bands
            self.tv_bounds.append(cert["EJ2"] / xcut**2)             # Chebyshev under the TARGET law P
        sel = retrospective_select(orc, env, rng, budget=self.budget, max_factory_coins=self.max_factory_coins)
        self.diagnostics.append(sel.diagnostics)
        return P2BridgeRecord(TargetBridge(sel), s, t)


from ..correction.abstract import LogEnvelope as _LE


class _TruncatedEnvelope(_LE):
    def __init__(self, Mc, kmin, kmax):
        super().__init__(Mc)
        self.kmin, self.kmax = kmin, kmax

    def sample_k(self, rng):
        while True:
            k = super().sample_k(rng)
            if self.kmin <= k <= self.kmax:
                return k
