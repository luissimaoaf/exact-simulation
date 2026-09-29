"""Paper 1 §10 "additive noise needs only the first level": guided bridge for  dX=(b(t,X)+.)dt+Sigma dW.

Covariance-matched guide (a=Sigma Sigma^T constant so A=a a_T^{-1}=I):
        dX = (b(t,X) - (X-v)/(T-t)) dt + Sigma dW,            X_0=u, X_T=v.
Volterra form (variation of constants):
        X_t = l_t + B_t + (T-t) int_0^t b(s,X_s)/(T-s) ds,     l_t=u+(v-u)t/T,  B_t=Sigma beta_t,
with beta a standard m-dim Brownian bridge on [0,T] (same law as (T-t)int dW/(T-s)).

Stability.  Because the kernel (T-t)/(T-s) <= 1 and b is L-Lipschitz, plain Gronwall gives
        sup_t |X_t - X~_t| <= e^{L T} ‖B - B~‖_inf                (no weights, no Levy area, no alpha>1/3),
which is the alpha->0 case of Paper 1's weighted estimate  ‖X-X~‖_gamma <= e^{LT}‖B-B~‖_gamma.

Error decomposition returned separately (Paper 1 Remark 7.2):
  driver   : e^{LT}‖Sigma‖ r_n, r_n the tail-first certified sup-error of the retained bridge polygon (rigorous in
             ideal arithmetic);
  solver   : error of the RK4 solve of the polygonal-driver Volterra equation on [0,T-h] -- ESTIMATED by step
             doubling (NOT certified; documented weakening of the interval verifier);
  terminal : 2 M h^q (pinning modulus of the latent path, computed from the actual last-cell nodes and tail-first
             per-level hat bounds) -- rigorous in ideal arithmetic (Paper 1 Alg. 1: E=E_pre+2M h^alpha).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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

    def G_enclosure(self, t0, t1, xc, r, v, T) -> tuple[float, float]:
        """Enclosure of G(t,x) = -b(t,x)^T S (x-v)/(T-t) over t in [t0,t1], |x-xc|<=r (b time-independent,
        Lipschitz L): |G-G(xc)| <= (1/tau)[L‖S‖|y|r + |b(xc)|‖S‖r + L‖S‖ r^2]."""
        tau_lo, tau_hi = T - t1, T - t0
        bc = self.b(0.5 * (t0 + t1), xc)
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
        nrm = np.linalg.norm(self.values, axis=1)   # polygon: norm maximised at vertices
        return max(nrm.max() - self.radius, 0.0), nrm.max() + self.radius


class AdditiveGuidedBridge:
    """Retained latent guided bridge X° ~ Q^{u,v} (additive noise) with nested certified enclosures."""

    def __init__(self, model: AdditiveModel, u, v, T: float, rng: np.random.Generator,
                 q_pin: float = 0.45, cut_extra: int = 10, substeps: int = 1, estimate_solver: bool = True, fast: bool = True,
                 certification: str = "certified", calib: float = 0.1, budget: Budget = NULL_BUDGET):
        assert 0 < q_pin < 0.5
        self.model, self.T = model, float(T)
        self.u, self.v = np.asarray(u, float), np.asarray(v, float)
        self.q_pin, self.cut_extra, self.substeps = q_pin, cut_extra, substeps
        self.estimate_solver = estimate_solver
        self.fast = fast
        assert certification in ("certified", "diagnostic")
        self.certification, self.calib = certification, calib
        self.budget = budget
        self.record = BrownianRecord(T, model.m, rng, a=np.zeros(model.m), b=np.zeros(model.m))   # bridge 0->0
        self._cache: dict = {}

    # ---------------------------------------------------------------- certified quantities
    def driver_radius(self, n: int) -> float:
        m = self.model
        return math.exp(m.L * self.T) * m.sig_norm * self.record.radius(n)

    def state_bound(self, n: int) -> float:
        """Rigorous sup|X°| (Gronwall for the Volterra equation; kernel <=1)."""
        m = self.model
        poly = self.record.nodes(n)
        sup_beta = np.linalg.norm(poly, axis=1).max() + self.record.radius(n)
        K0 = max(np.linalg.norm(self.u), np.linalg.norm(self.v)) + m.sig_norm * sup_beta
        return (K0 + m.B0 * self.T) * math.exp(m.L * self.T)

    def pinning_constant(self, n: int, q: int) -> float:
        """M with |X°_t - v| <= M (T-t)^{q_pin} on [T-h,T), h=T 2^-q, q<=n.

        X_t - v = (T-t)[(u-v)/T + int_0^t b/(T-s) ds] + Sigma beta_t and
        |beta_t| <= |beta~_n(t)| + endpoint_modulus(T-t, n)   (per-level hat bounds, tail-first thresholds)."""
        m, T, qp = self.model, self.T, self.q_pin
        h, hn = T * 2.0 ** (-q), T * 2.0 ** (-n)
        beta = self.record.nodes(n)
        norms = np.linalg.norm(beta, axis=1)
        N = 2 ** n
        poly = 0.0
        if q <= n:
            j0 = N - 2 ** (n - q)
            js = np.arange(j0, N - 1)
            if len(js):
                smin = T - (js + 1) * hn
                poly = float(np.max(np.maximum(norms[js], norms[js + 1]) / smin**qp))
            poly = max(poly, float(norms[N - 1] * hn ** (1 - qp) / hn))     # last cell: linear to 0 at T
        else:
            poly = float(norms[N - 1] * h ** (1 - qp) / hn)                  # cutoff inside the last cell
        # hat-tail part, breakpoints s = h 2^-j
        om = 0.0
        for j in range(0, 200):
            s = h * 2.0 ** (-j)
            om = max(om, self.record.endpoint_modulus(s, n) / s**qp)
        supb = m.B0 + m.L * self.state_bound(n)
        smax_term = max(h ** (1 - qp) * math.log(max(T / h, math.e)) if h < T * math.exp(-1 / (1 - qp)) else
                        max((h * 2.0 ** (-j)) ** (1 - qp) * math.log(T / (h * 2.0 ** (-j))) for j in range(0, 200)), 0.0)
        return float(np.linalg.norm(self.u - self.v) / T * h ** (1 - qp) + supb * smax_term
                     + m.sig_norm * (poly + om))

    def tail_integral_bound(self, n: int, q: int) -> float:
        """Rigorous bound of |int_{T-h}^T G(t,X°_t) dt| for G=-b^T S (x-v)/(T-t) (constant covariance):
        |G| <= supb ‖S‖ M (T-t)^{q_pin-1}  =>  supb ‖S‖ M h^{q_pin}/q_pin   (Paper 2 eq. (45))."""
        m = self.model
        h = self.T * 2.0 ** (-q)
        M = self.pinning_constant(n, q)
        supb = m.B0 + m.L * self.state_bound(n)
        val = supb * m.S_norm * M * h ** self.q_pin / self.q_pin
        return val if self.certification == "certified" else self.calib * val

    # ---------------------------------------------------------------- numerical center
    def _solve_center(self, n: int, q: int, sub: int) -> tuple[np.ndarray, np.ndarray]:
        """RK4 for Y'=b(t,l_t+B~_t+(T-t)Y)/(T-t) on nodes of level n up to T-h_q; returns times, X~ values."""
        m, T = self.model, self.T
        beta = self.record.nodes(n)
        Bp = beta @ m.Sigma.T
        N = 2 ** n
        Nend = N - 2 ** (n - q) if q <= n else N - 1
        ts = np.linspace(0.0, T, N + 1)
        u, v = self.u, self.v
        Y = np.zeros(m.d)
        Xs = np.empty((Nend + 1, m.d))
        Xs[0] = u
        b = m.b

        def rhs(t, Y, Bt):
            return b(t, u + (v - u) * t / T + Bt + (T - t) * Y) / (T - t)

        def rk4(ta, dt, Y, Ba, slope):
            Bm = Ba + slope * dt / 2
            Bb = Ba + slope * dt
            k1 = rhs(ta, Y, Ba)
            k2 = rhs(ta + dt / 2, Y + dt / 2 * k1, Bm)
            k3 = rhs(ta + dt / 2, Y + dt / 2 * k2, Bm)
            k4 = rhs(ta + dt, Y + dt * k3, Bb)
            return Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        for i in range(Nend):
            t0, t1 = ts[i], ts[i + 1]
            dt = (t1 - t0) / sub
            slope = (Bp[i + 1] - Bp[i]) / (t1 - t0)
            for s in range(sub):
                ta = t0 + s * dt
                Y = rk4(ta, dt, Y, Bp[i] + slope * (ta - t0), slope)
            self.budget.tick("rk4_cell")
            Xs[i + 1] = u + (v - u) * t1 / T + Bp[i + 1] + (T - t1) * Y
        tt = list(ts[: Nend + 1])
        if q > n:                                       # geometric grid inside the last polygon cell
            hn = T * 2.0 ** (-n)
            hcut = T * 2.0 ** (-q)
            slope = (Bp[N] - Bp[N - 1]) / hn
            sc = T - ts[N - 1]
            ratio = 0.8 ** (1.0 / sub)
            extra_t, extra_X = [], []
            while sc > hcut * (1 + 1e-9):
                sN = max(sc * ratio, hcut)
                dt = sc - sN
                t = T - sc
                Y = rk4(t, dt, Y, Bp[N - 1] + slope * (t - ts[N - 1]), slope)
                sc = sN
                t = T - sc
                extra_t.append(t)
                extra_X.append(u + (v - u) * t / T + (Bp[N - 1] + slope * (t - ts[N - 1])) + sc * Y)
                self.budget.tick("rk4_cell")
            tt += extra_t
            Xs = np.vstack([Xs] + [np.array(extra_X)]) if extra_X else Xs
        return np.array(tt), Xs


    # ---------------------------------------------------------------- fast exact-affine center (linear b)
    def _solve_center_linear(self, n: int, q: int, ngl: int) -> tuple[np.ndarray, np.ndarray]:
        """Same center as `_solve_center` for b(t,x)=Bx+c, via cellwise affine maps (Y'=BY+F(t), F rational-linear):
        Y_{i+1}=e^{BD}Y_i+g_i, g_i by Gauss-Legendre (ngl nodes) vectorised over cells; the recurrence with the common
        matrix e^{BD} is solved in the eigenbasis with scipy lfilter.  Geometric substeps inside the last cell."""
        from scipy.linalg import expm
        from scipy.signal import lfilter
        ou = self.model._ou
        Bm, cc = ou.B, ou.c
        m, T = self.model, self.T
        Bp = self.record.nodes(n) @ m.Sigma.T
        N = 2 ** n
        Nend = N - 2 ** (n - q) if q <= n else N - 1
        hn = T / N
        xg, wg = np.polynomial.legendre.leggauss(ngl)
        xg, wg = 0.5 * (xg + 1), 0.5 * wg
        u, v = self.u, self.v
        ts = np.arange(Nend + 1) * hn
        # dyadic cells 0..Nend-1
        t0 = ts[:-1]
        s = t0[:, None] + hn * xg[None, :]                                    # (Nend, ngl)
        Bs = Bp[:Nend, None, :] + (Bp[1:Nend + 1] - Bp[:Nend])[:, None, :] * xg[None, :, None]
        ell = u[None, None, :] + (v - u)[None, None, :] * s[..., None] / T
        F = (np.einsum("ij,klj->kli", Bm, ell + Bs) + cc) / (T - s)[..., None]
        E = np.array([expm(Bm * hn * (1 - x)) for x in xg])                   # (ngl,d,d)
        g = hn * np.einsum("g,gij,kgj->ki", wg, E, F)                         # (Nend, d)
        Phi = expm(Bm * hn)
        lam, V = np.linalg.eig(Phi)
        Vi = np.linalg.inv(V)
        w = g @ Vi.T                                                           # (Nend,d) complex
        z = np.stack([lfilter([1.0], [1.0, -lam[j]], w[:, j]) for j in range(m.d)], axis=1)
        Y = np.vstack([np.zeros((1, m.d)), (z @ V.T).real])                    # Y_0..Y_Nend
        Bnode = Bp[:Nend + 1]
        Xs = u[None, :] + (v - u)[None, :] * ts[:, None] / T + Bnode + (T - ts)[:, None] * Y
        tt = list(ts)
        if q > n:
            hcut = T * 2.0 ** (-q)
            slope = (Bp[N] - Bp[N - 1]) / hn
            sc = T - ts[Nend]                       # distance to T tracked directly (no cancellation)
            Yc = Y[-1].copy()
            ratio = 0.8
            ex_t, ex_X = [], []
            while sc > hcut * (1 + 1e-9):
                sN = max(sc * ratio, hcut)
                D = sc - sN
                t = T - sc
                sg = t + D * xg
                Bsg = Bp[N - 1][None, :] + slope[None, :] * (sg - ts[Nend])[:, None]
                Fg = (np.einsum("ij,gj->gi", Bm, u + (v - u) * sg[:, None] / T + Bsg) + cc) / (T - sg)[:, None]
                Eg = np.array([expm(Bm * D * (1 - x)) for x in xg])
                gg = D * np.einsum("g,gij,gj->i", wg, Eg, Fg)
                Yc = expm(Bm * D) @ Yc + gg
                sc = sN
                t = T - sc
                ex_t.append(t)
                ex_X.append(u + (v - u) * t / T + Bp[N - 1] + slope * (t - ts[Nend]) + sc * Yc)
            tt += ex_t
            if ex_X:
                Xs = np.vstack([Xs, np.array(ex_X)])
        return np.array(tt), Xs

    def enclosure(self, n: int, q: int | None = None) -> Enclosure:
        """Certified enclosure at driver level n (terminal cutoff h=T 2^-q, default q=n+cut_extra)."""
        q = n + self.cut_extra if q is None else q
        key = (n, q)
        if key in self._cache:
            return self._cache[key]
        m, T = self.model, self.T
        if getattr(self.model, "_ou", None) is not None and self.fast:
            tt, Xc = self._solve_center_linear(n, q, 6)
            if self.estimate_solver:
                tt2, X2 = self._solve_center_linear(n, q, 10)
                e_solver = float(max(np.linalg.norm(Xc - X2, axis=1).max(), 0.0)) + 1e-15
                Xc, tt = X2, tt2
            else:
                e_solver = 0.0
        else:
            tt, Xc = self._solve_center(n, q, self.substeps)
            if self.estimate_solver:
                tt2, X2 = self._solve_center(n, q, 2 * self.substeps)
                k0 = (2 ** n - 2 ** (n - q) if q <= n else 2 ** n - 1) + 1          # common dyadic nodes
                diffs = [np.linalg.norm(Xc[:k0] - X2[:k0], axis=1).max(), np.linalg.norm(Xc[-1] - X2[-1])]
                e_solver = float(max(diffs)) * (16.0 / 15.0) + 1e-15                 # Richardson-type estimate
                Xc, tt = X2, tt2
            else:
                e_solver = 0.0
        h = T * 2.0 ** (-q)
        times = np.concatenate([tt, [T]])
        vals = np.vstack([Xc, self.v[None, :]])
        M = self.pinning_constant(n, q)
        e_tail = 2.0 * M * h ** self.q_pin
        e_drv = self.driver_radius(n)
        if self.certification == "certified":
            status = "certified (ideal arithmetic; solver error estimated)"
        else:
            # UNCERTIFIED diagnostic: the certified formulas x calibration factor (the certificate is ~10-30x
            # conservative; monotone in n).  Flagged in every output; never used for exactness claims.
            e_drv, e_tail = self.calib * e_drv, self.calib * e_tail
            status = "diagnostic-UNCERTIFIED (calibrated certificate)"
        radius = e_drv + e_solver + e_tail
        enc = Enclosure(times, vals, radius, dict(driver=e_drv, solver=e_solver, terminal=e_tail, h=h, M=M,
                                                  pre=e_drv + e_solver, status=status), n)
        self._cache[key] = enc
        return enc

    def max_norm_interval(self, n: int) -> tuple[float, float]:
        return self.enclosure(n).sup_norm_interval()

    def refine_to(self, eps: float, n0: int = 4, nmax: int = 22) -> Enclosure:
        """Adaptive refinement (Algorithm 1 analogue): increase the level of the SAME latent path until radius<=eps."""
        n = n0
        enc = None
        while n <= nmax:
            enc = self.enclosure(n)
            if enc.radius <= eps:
                return enc
            n += 1
        raise RuntimeError(f"tolerance {eps} not reached by level {nmax}; last radius {enc.radius:.3g}, parts {enc.parts}")
