"""Paper 1 §10 "additive noise needs only the first level": guided bridge for  dX=(b(t,X)+.)dt+Sigma dW.

Covariance-matched guide (a=Sigma Sigma^T constant so A=a a_T^{-1}=I):
        dX = (b(t,X) - (X-v)/(T-t)) dt + Sigma dW,            X_0=u, X_T=v.
Volterra form (variation of constants):
        X_t = l_t + B_t + (T-t) int_0^t b(s,X_s)/(T-s) ds,     l_t=u+(v-u)t/T,  B_t=Sigma beta_t,
with beta a standard m-dim Brownian bridge on [0,T] (same law as (T-t)int dW/(T-s)).  Weighted stability
(b Lipschitz, constant L):
        sup_t |X_t-X~_t|/(T-t)^gamma <= e^{L T} ||B-B~||_gamma       (0<=gamma<=1),
so  ||X-X~||_inf <= T^gamma e^{LT} ||B - B~||_gamma.   Only the retained Brownian-bridge record is needed; no
Levy area, no alpha>1/3 restriction.

Error decomposition returned separately (Paper 1 Remark 7.2):
  driver   : rigorous (ideal arithmetic) from the tail-first certificate of the bridge record;
  solver   : numerical error of the RK4 solve of the polygonal-driver Volterra equation  -- ESTIMATED by step
             doubling (NOT certified; documented weakening of the interval verifier);
  terminal : rigorous bound of the splice on [T-h,T] from sup|b| (Gronwall) and Holder pinning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..brownian.record import BrownianRecord
from ..rng import NULL_BUDGET, Budget


@dataclass
class AdditiveModel:
    """dX = b(t,X)dt + Sigma dW with b globally Lipschitz in x: |b(t,x)-b(t,y)|<=L|x-y|, |b(t,0)|<=B0."""

    Sigma: np.ndarray
    b: Callable[[float, np.ndarray], np.ndarray]
    L: float
    B0: float

    def __post_init__(self):
        self.Sigma = np.atleast_2d(np.asarray(self.Sigma, float))
        self.d, self.m = self.Sigma.shape
        self.a = self.Sigma @ self.Sigma.T
        self.S = np.linalg.inv(self.a)
        self.sig_norm = float(np.linalg.norm(self.Sigma, 2))
        self.S_norm = float(np.linalg.norm(self.S, 2))

    @classmethod
    def from_ou(cls, ou) -> "AdditiveModel":
        B, c = ou.B, ou.c
        m = cls(ou.Sigma, lambda t, x: B @ x + c, float(np.linalg.norm(B, 2)), float(np.linalg.norm(c)))
        m._ou = ou
        return m

    # enclosure of G(t,x) = -b(t,x)^T S (x-v)/(T-t) over t in [t0,t1], |x-xc|<=r   (OU: b affine)
    def G_enclosure(self, t0, t1, xc, r, v, T) -> tuple[float, float]:
        tau_lo, tau_hi = T - t1, T - t0
        bc = self.b(0.5 * (t0 + t1), xc)      # b time-independent for OU; general b needs time modulus
        y = xc - v
        core = -float(bc @ self.S @ y)
        dev = (self.L * self.S_norm * np.linalg.norm(y) * r + np.linalg.norm(bc) * self.S_norm * r
               + self.L * self.S_norm * r * r)
        lo, hi = core - dev, core + dev
        cands = [lo / tau_lo, lo / tau_hi, hi / tau_lo, hi / tau_hi]
        return min(cands), max(cands)


@dataclass
class Enclosure:
    """Finite-description continuous path with certified uniform radius (contract D1/D3)."""

    times: np.ndarray
    values: np.ndarray
    radius: float
    parts: dict
    level: int

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return np.stack([np.interp(t, self.times, self.values[:, j]) for j in range(self.values.shape[1])], axis=-1)

    def sup_norm_interval(self) -> tuple[float, float]:
        nrm = np.linalg.norm(self.values, axis=1)   # polygon: sup at vertices
        return max(nrm.max() - self.radius, 0.0), nrm.max() + self.radius


class AdditiveGuidedBridge:
    """Retained latent guided bridge X° ~ Q^{u,v} (additive noise) with nested certified enclosures."""

    def __init__(self, model: AdditiveModel, u, v, T: float, rng: np.random.Generator,
                 abar: float = 0.45, gamma: float = 0.30, q_pin: float = 0.4, substeps: int = 2,
                 budget: Budget = NULL_BUDGET):
        assert gamma < abar <= 0.5 and q_pin <= abar
        self.model, self.T = model, float(T)
        self.u, self.v = np.asarray(u, float), np.asarray(v, float)
        self.abar, self.gamma, self.q_pin, self.substeps = abar, gamma, q_pin, substeps
        self.budget = budget
        self.record = BrownianRecord(T, model.m, rng, a=np.zeros(model.m), b=np.zeros(model.m))   # bridge 0->0
        self._cache: dict = {}
        self._Hb = None

    # ---------------------------------------------------------------- certified driver quantities
    def hb(self) -> float:
        if self._Hb is None:
            self._Hb = self.record.holder_constant(self.abar)
        return self._Hb

    def driver_weighted_norm(self, n: int) -> float:
        """Certified ||beta-beta~_n||_gamma (sup |diff|/(T-t)^gamma)."""
        hn = self.T * 2.0 ** (-n)
        return max(self.record.radius(n) * hn ** (-self.gamma), 2.0 * self.hb() * hn ** (self.abar - self.gamma))

    def driver_radius(self, n: int) -> float:
        m = self.model
        return self.T ** self.gamma * math.exp(m.L * self.T) * m.sig_norm * self.driver_weighted_norm(n)

    def state_bound(self, n: int) -> float:
        """Rigorous sup|X°| (Gronwall for the Volterra equation; kernel <=1)."""
        m = self.model
        poly = self.record.nodes(n)
        sup_beta = np.linalg.norm(poly, axis=1).max() + self.record.radius(n)
        K0 = max(np.linalg.norm(self.u), np.linalg.norm(self.v)) + m.sig_norm * sup_beta
        return (K0 + m.B0 * self.T) * math.exp(m.L * self.T)

    def pinning_constant(self, n: int) -> tuple[float, float]:
        """(M,q): |X°_t - v| <= M (T-t)^q for all t<T  (rigorous, see module doc)."""
        m, T, q = self.model, self.T, self.q_pin
        supb = m.B0 + m.L * self.state_bound(n)
        M = (np.linalg.norm(self.u - self.v) * T ** (-q) + m.sig_norm * self.hb() * T ** (self.abar - q)
             + supb * T ** (1 - q) / (math.e * (1 - q)))
        return float(M), q

    def tail_integral_bound(self, n: int, q: int) -> float:
        """Rigorous bound of |int_{T-h}^T G(t,X°_t) dt|, h=T2^-q, for G=-b^T S (x-v)/(T-t):
        |G| <= supb ‖S‖ M (T-t)^{q_pin-1}  =>  supb ‖S‖ M h^{q_pin}/q_pin   (Paper 2 eq. (45), constant covariance)."""
        m = self.model
        M, qq = self.pinning_constant(n)
        supb = m.B0 + m.L * self.state_bound(n)
        h = self.T * 2.0 ** (-q)
        return supb * m.S_norm * M * h ** qq / qq

    # ---------------------------------------------------------------- numerical center
    def _solve_center(self, n: int, q: int, sub: int) -> tuple[np.ndarray, np.ndarray]:
        """RK4 for Y'=b(t,l_t+B~_t+(T-t)Y)/(T-t) on grid nodes of level n up to T-h_q; returns times, X~ values."""
        m, T = self.model, self.T
        beta = self.record.nodes(n)                    # polygon of the bridge, (2^n+1, m)
        Bp = beta @ m.Sigma.T                          # (2^n+1, d)
        N = 2 ** n
        Nend = N - 2 ** (n - q)                        # last node index (time T - T 2^-q)
        ts = np.linspace(0.0, T, N + 1)
        ell = lambda t: self.u + (self.v - self.u) * t / T
        Y = np.zeros(m.d)
        Xs = [self.u.copy()]
        tt = [0.0]

        def rhs(t, Y, Bt):
            x = ell(t) + Bt + (T - t) * Y
            return m.b(t, x) / (T - t)

        for i in range(Nend):
            t0, t1 = ts[i], ts[i + 1]
            dt = (t1 - t0) / sub
            slope = (Bp[i + 1] - Bp[i]) / (t1 - t0)
            for s in range(sub):
                ta = t0 + s * dt
                Ba = Bp[i] + slope * (ta - t0)
                Bm = Ba + slope * dt / 2
                Bb = Ba + slope * dt
                k1 = rhs(ta, Y, Ba)
                k2 = rhs(ta + dt / 2, Y + dt / 2 * k1, Bm)
                k3 = rhs(ta + dt / 2, Y + dt / 2 * k2, Bm)
                k4 = rhs(ta + dt, Y + dt * k3, Bb)
                Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                self.budget.tick("rk4_step")
            t = ts[i + 1]
            Xs.append(ell(t) + Bp[i + 1] + (T - t) * Y)
            tt.append(t)
        return np.array(tt), np.array(Xs)

    def enclosure(self, n: int, q: int | None = None) -> Enclosure:
        """Certified enclosure at driver level n (terminal cutoff h=T 2^-q, default q=n-2... see below)."""
        q = max(1, n - 3) if q is None else q
        key = (n, q)
        if key in self._cache:
            return self._cache[key]
        m, T = self.model, self.T
        tt, X1 = self._solve_center(n, q, self.substeps)
        _, X2 = self._solve_center(n, q, 2 * self.substeps)
        e_solver = float(np.max(np.linalg.norm(X1 - X2, axis=1))) * (16.0 / 15.0) + 1e-15   # Richardson estimate
        Xc = X2
        h = T * 2.0 ** (-q)
        # terminal splice: linear to v on [T-h,T]; rigorous bound 2 M h^q  (Paper 1 Alg.1 line E = Epre + 2Mh^alpha)
        M, qq = self.pinning_constant(n)
        e_tail = 2.0 * M * h ** qq
        times = np.concatenate([tt, [T]])
        vals = np.vstack([Xc, self.v[None, :]])
        e_drv = self.driver_radius(n)
        radius = e_drv + e_solver + e_tail
        enc = Enclosure(times, vals, radius, dict(driver=e_drv, solver=e_solver, terminal=e_tail, h=h, M=M), n)
        self._cache[key] = enc
        return enc

    def max_norm_interval(self, n: int) -> tuple[float, float]:
        return self.enclosure(n).sup_norm_interval()

    def refine_to(self, eps: float, n0: int = 4, nmax: int = 22) -> Enclosure:
        """Adaptive refinement (Algorithm 1 analogue): increase the level of the SAME latent path until radius<=eps."""
        n = n0
        while n <= nmax:
            enc = self.enclosure(n)
            if enc.radius <= eps:
                return enc
            n += 1
        raise RuntimeError(f"tolerance {eps} not reached by level {nmax}; last radius {enc.radius:.3g}, parts {enc.parts}")
