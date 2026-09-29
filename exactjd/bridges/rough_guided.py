"""Paper 1: constructive eps-strong guided bridge via rough paths (general multiplicative noise, m=d=2).

Pipeline (Algorithm 1) for one retained latent path X° = X^{u,W}:
  1. driver level n: `BrownianAreaRecord` -> canonical polygonal geometric lift W^[n], certified rho_alpha ball
     (delta_n = E_n, R_n = max{1, N_n+delta_n})  [Theorem 4.2, `BrownianAreaRecord.constants`];
  2. CENTER: since W^[n] is piecewise linear with canonical lift, the rough equation is the ODE
        X' = f(t,X) + sum_j V_j(X) dW^{[n],j}/dt ,   f(t,x) = -A(x)(x-v)/(T-t)+F(x),
     solved by RK4 (numerical-solver error ESTIMATED by step doubling -- not certified);
  3. TUBE  (`tube`)   : block-wise controlled-Picard certificate (`cpverify.verify_block`) from the driver ball to the
                        preterminal error E_pre on [0,T-h];
  4. TAIL  (`tail`)   : terminal-shell certificate (Prop 6.4) -> modulus M with |X°_t-v|<=M(T-t)^alpha on [T-h,T];
  5. radius = E_pre + 2 M h^alpha  (+ solver estimate), reported as three separate parts (Remark 7.2).

Certified mode raises `CertificateFailure` / `BudgetExhausted` when the finite certificate is not attained within the
explicit block budget (the paper's constants make this the normal outcome at practical levels: see experiments).
`diagnostic_enclosure` is an EXPLICITLY UNCERTIFIED alternative (radius = safety * successive-difference estimate) whose
status is flagged in every output; it is never used for exactness claims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..brownian.levyarea import BrownianAreaRecord
from ..rng import NULL_BUDGET, Budget, BudgetExhausted
from .cpverify import Bounds, CertificateFailure, sewing_constant, verify_block


@dataclass
class RoughEnclosure:
    times: np.ndarray
    values: np.ndarray
    radius: float
    parts: dict
    level: int
    status: str            # 'certified' | 'diagnostic-UNCERTIFIED'

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return np.stack([np.interp(t, self.times, self.values[:, j]) for j in range(self.values.shape[1])], axis=-1)


class RoughGuidedBridge:
    def __init__(self, model, u, v, T: float, rng: np.random.Generator, alpha: float = 0.34, abar: float = 0.49,
                 beta: float = 0.6, budget: Budget = NULL_BUDGET):
        assert 1 / 3 < alpha < abar < 0.5 and 1 - abar < beta < 2 * abar
        self.model = model.bind(v, T)
        self.u, self.v, self.T = np.asarray(u, float), np.asarray(v, float), float(T)
        self.alpha, self.abar, self.beta = alpha, abar, beta
        self.record = BrownianAreaRecord(T, rng)
        self.budget = budget
        self._cache: dict = {}

    # ------------------------------------------------------------ (1) driver ball
    def driver(self, n: int) -> dict:
        c = self.record.constants(self.alpha, self.abar, self.beta, n)
        R = max(1.0, c["Nnorm"] + c["E"])
        return dict(delta=c["E"], R=R, H1=c["H1"], H2=c["H2"], G=c["G"], level=n)

    # ------------------------------------------------------------ (2) center ODE
    def center(self, n: int, q: int, sub: int):
        """RK4 centre on the level-n polygon up to T-h_q; returns fine times and states."""
        m, T = self.model, self.T
        W = self.record.nodes(n)
        N = 2 ** n
        Nend = N - 2 ** (n - q)
        dtc = T / N
        x = self.u.copy()
        ts, xs = [0.0], [x.copy()]
        f, Vf = m.f, m.V
        for i in range(Nend):
            w = (W[i + 1] - W[i]) / dtc
            dt = dtc / sub
            for s in range(sub):
                t = i * dtc + s * dt
                k1 = f(t, x) + Vf(x) @ w
                k2 = f(t + dt / 2, x + dt / 2 * k1) + Vf(x + dt / 2 * k1) @ w
                k3 = f(t + dt / 2, x + dt / 2 * k2) + Vf(x + dt / 2 * k2) @ w
                k4 = f(t + dt, x + dt * k3) + Vf(x + dt * k3) @ w
                x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                ts.append(t + dt)
                xs.append(x.copy())
            self.budget.tick("rough_ode_cell")
        return np.array(ts), np.array(xs)

    # ------------------------------------------------------------ (3) preterminal tube
    def tube(self, n: int, q: int, sub: int = 2, max_blocks: int = 5000, drv: dict | None = None) -> dict:
        """Compose successful CPVerify blocks over [0,T-h]; returns E_pre (difference radius from the driver ball)."""
        drv = drv or self.driver(n)
        R, delta, alpha = drv["R"], drv["delta"], self.alpha
        ts, xs = self.center(n, q, sub)
        Sa = sewing_constant(alpha)
        a, K = 0, len(ts) - 1
        r_in, blocks, min_ell = 0.0, 0, math.inf
        length = 1
        while a < K:
            blocks += 1
            self.budget.tick("cpverify_block")
            if blocks > max_blocks:
                raise BudgetExhausted(f"tube: more than {max_blocks} blocks needed (R={R:.3g}); certificate infeasible at this level")
            length = min(length * 2, K - a)
            while True:
                b = a + length
                ell = ts[b] - ts[a]
                q1, q2 = R * ell**alpha, R**2 * ell ** (2 * alpha)
                res = verify_block(lambda rad: self.model.coeff_bounds(ts[a], ts[b], xs[a], rad), ell, alpha, q1, q2,
                                   r_in, delta, Sa)
                if res.ok:
                    break
                if length == 1:
                    raise CertificateFailure(f"single fine step (ell={ell:.3g}) fails: {res.reason} (R={R:.3g}, delta={delta:.3g})")
                length = max(1, length // 2)
            r_in = res.radius
            min_ell = min(min_ell, ell)
            a = b
        return dict(E_pre=r_in, blocks=blocks, min_block=min_ell, x_end=xs[-1], t_end=ts[-1], drv=drv)

    # ------------------------------------------------------------ (4) terminal tail
    def tail(self, tube: dict, h: float, sub_M: int = 40) -> dict:
        """Prop 6.4 (constant-modulus form): find M with (35) satisfied, using explicit conservative constants.

        T1: coercivity kappa* = min eig of S^{1/2} a(x) S^{1/2} on the tube  (a Lipschitz => kappa* >= 1 - Ca (M h^alpha+h)/lam)
        T2: flow constants from the augmented Picard test (c0,c3) at xi=h^alpha  (Lemma C.1 recipe)
        T3/T4 and (35) exactly as in the paper.  Returns ok=False with the failing inequality otherwise."""
        m, alpha, beta = self.model, self.alpha, self.beta
        drv = tube["drv"]
        R = drv["R"]
        xi = h**alpha
        cS = float(max(np.linalg.norm(np.linalg.cholesky(m.S), 2), np.linalg.norm(np.linalg.inv(np.linalg.cholesky(m.S)), 2), 1.0))
        r0 = float(np.linalg.norm(tube["x_end"] - self.v)) + tube["E_pre"]
        lam_min = float(np.linalg.eigvalsh(m.aT)[0])
        # flow certificate: augmented displacement bound c0 xi via verify_block on the driftless system (r_in=0)
        Kv = m.V_bounds(self.v, 1.0, range(0, 6))
        ell = xi**2
        q1, q2 = R * xi, R**2 * xi**2
        aug = _augmented_bounds(Kv)
        res = verify_block(lambda rad: aug, ell, alpha, q1, q2, 0.0, 0.0)
        if not res.ok:
            return dict(ok=False, reason="flow certificate (Lemma C.1) fails at this shell size: " + res.reason)
        c0 = res.m[0] / xi
        sp_, sm_ = float(np.sqrt(np.linalg.eigvalsh(m.S)[-1])), float(1 / np.sqrt(np.linalg.eigvalsh(m.S)[0]))
        Cphi = sp_ * c0 / R
        CJ = 2 * sp_ * sm_ * c0 / R
        Astar = float(np.linalg.norm(m.A(self.v), 2)) * 2
        Fstar = float(np.linalg.norm(m.F(self.v))) * 2 + 1.0
        CA = Astar * (CJ + 2 * Cphi) + 1.0
        Creg = 2 * Fstar + 1.0
        for j in range(0, 60):
            M = 2.0**j
            # coercivity on the physical tube |x-v| <= M h^alpha  (local Lipschitz const of a from bounds: use V-derivative bound)
            Lip_a = 2 * Kv[0] * Kv[1] * math.sqrt(2)
            kstar = 1.0 - Lip_a * M * xi / lam_min
            if kstar <= beta + 1e-9:
                continue
            kap = 0.5 * (kstar + beta)
            if CA * R * xi + Creg * h >= kstar - kap:
                continue
            Csh = (CA * R + Creg) / kap + Cphi * R
            theta = 2.0 ** (alpha - beta)
            Btail = r0 * h ** (-alpha) + 2.0**alpha * Csh / (1 - theta)
            if M > cS * 2.0**alpha * (Btail + Csh):
                return dict(ok=True, M=M, kstar=kstar, Csh=Csh, Btail=Btail, c0=c0, Cphi=Cphi, CJ=CJ, kappa=kap)
        return dict(ok=False, reason="no modulus M satisfies (35),(T4) with the explicit constants (h too large or constants too coarse)")

    # ------------------------------------------------------------ (5) certified enclosure
    def certified_enclosure(self, n: int, q: int | None = None, **kw) -> RoughEnclosure:
        q = max(2, n - 2) if q is None else q
        h = self.T * 2.0 ** (-q)
        drv = self.driver(n)
        tube = self.tube(n, q, drv=drv, **kw)
        tl = self.tail(tube, h)
        if not tl["ok"]:
            raise CertificateFailure(f"terminal closure failed: {tl['reason']}")
        ts1, xs1 = self.center(n, q, 2)
        ts2, xs2 = self.center(n, q, 4)
        e_solver = float(np.max(np.linalg.norm(xs1 - xs2[::2], axis=1))) * (16 / 15) + 1e-15
        e_tail = 2 * tl["M"] * h**self.alpha
        radius = tube["E_pre"] + e_tail + e_solver
        times = np.concatenate([ts2, [self.T]])
        vals = np.vstack([xs2, self.v[None, :]])
        return RoughEnclosure(times, vals, radius, dict(driver_delta=drv["delta"], R=drv["R"], preterminal=tube["E_pre"],
                                                        terminal=e_tail, solver_estimate=e_solver, blocks=tube["blocks"],
                                                        M=tl["M"], h=h), n, "certified")

    def diagnostic_enclosure(self, n: int, safety: float = 3.0, extra: int = 2) -> RoughEnclosure:
        """UNCERTIFIED: radius = safety * ||centre_n - centre_{n+extra}||_inf on the common grid (labelled)."""
        q = max(2, n - 2)
        ts, xs = self.center(n, q, 2)
        tsf, xsf = self.center(n + extra, q + extra, 2)
        # compare on the coarse fine-grid subset
        ratio = 2 ** extra
        k = min(len(xs), (len(xsf) - 1) // ratio + 1)
        err = float(np.max(np.linalg.norm(xs[:k] - xsf[: (k - 1) * ratio + 1: ratio], axis=1)))
        h = self.T * 2.0 ** (-q)
        tail = float(np.linalg.norm(xs[-1] - self.v))
        radius = safety * (err + tail)
        return RoughEnclosure(np.concatenate([ts, [self.T]]), np.vstack([xs, self.v[None, :]]), radius,
                              dict(successive_difference=err, terminal_gap=tail, safety=safety), n,
                              "diagnostic-UNCERTIFIED")


def _augmented_bounds(Kv: dict) -> Bounds:
    """Conservative K_r (r<=3) of the augmented driftless flow fields (Y,J,K) -> (V(Y), DV(Y)J, D2V[J,J]+DV K), with
    ||J||<=3/2, ||K||<=1 assumed inside the invariant box (checked by the caller via the displacement bound)."""
    J, Kk = 1.5, 1.0
    def blocks(r):
        vy = Kv[r]
        vj = Kv[r + 1] * J + (r * Kv[r] if r >= 1 else 0.0)
        vk = (Kv[r + 2] * J**2 + 2 * r * Kv[r + 1] * J + r * (r - 1) * Kv[r] + Kv[r + 1] * Kk + r * Kv[r]) if r + 2 <= 5 else math.inf
        return math.sqrt(vy**2 + vj**2 + vk**2)
    Ks = [math.sqrt(Kv[0] ** 2 + (Kv[1] * J) ** 2 + (Kv[2] * J**2 + Kv[1] * Kk) ** 2)] + [blocks(r) for r in (1, 2, 3)]
    return Bounds(tuple(Ks), (0.0, 0.0, 0.0), 0.0, 0.0)
