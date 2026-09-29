"""Enhanced Brownian driver W=(W,W) in R^2 with certified aggregate Levy-area residual (Paper 1 §4, BCD).

Implementation of the *second-level information* of the retained hierarchy for m=2 (single Levy area).

Facts used (all exact identities, proved in the module comments / tests):
  * midpoint noise xi_c ~ N(0,I_2) of a level-k cell c (length delta_k, increment D_c over the cell);
  * Chen splitting of the Levy area of cell c into halves:
        a_c = a_L + a_R + (1/4) sqrt(delta_k) (xi^1 D^2 - xi^2 D^1);
    hence the Levy area of any dyadic interval I at depth i is  a_I = sum_{k>i} S_k(I),
    S_k(I)=sum_{level-k cells c in I} (1/4) sqrt(delta_k)(xi_c^1 D_c^2 - xi_c^2 D_c^1);
  * the aggregate residual of the level-n polygon over grid pair (p,q) is  sum_{k>n} S_k(p,q)  and equals
    (W - W^[n])^{12}_{p,q}  (Chen + equal first-level cell increments; Paper 1 eq. (18)).

Given the coarse levels, S_k(I) = sum_c s_c e_c with s_c=(1/4)sqrt(delta_k)|D_c| and e_c=xi_c . n_c ~ N(0,1) iid
(n_c = (D^2,-D^1)/|D|).  Tree-Gaussian decomposition (top-down, sigma_I^2 = sum s^2 over I):
        S_L = (sigma_L^2/sigma_I^2) S_I + (sigma_L sigma_R/sigma_I) eta_I,  S_R = S_I - S_L,  S_root = sigma_root eta_root
with N iid N(0,1) variables eta, and the orthogonal comp xi_perp_c = xi_c . D_c/|D_c| iid N(0,1).  Writing g_I=S_I/sigma_I:
        g_child^2 <= g_I^2 + eta_I^2   =>   |g_I| <= u sqrt(depth+1)  when all |eta|<=u.
Tail-first events are on the 2N iid variables (eta, xi_perp) with exactly known probability -> deterministic
bounds for all unrevealed levels.  Only m=2 is implemented (see CHECKLIST: m>2 needs joint conditioning over pairs).
"""
from __future__ import annotations

import math

import numpy as np

from . import tailfirst as tf
from .record import BrownianRecord

C_B = 2.0 / (1.0 - 2.0 ** -0.5)


class BrownianAreaRecord(BrownianRecord):
    """Free 2-d Brownian motion on [0,T] with certified Holder-alpha-bar first level, area Holder constant and
    aggregate residual constants."""

    def __init__(self, T: float, rng: np.random.Generator, gamma: float = 2.5):
        self.h, self.d, self.rng = float(T), 2, rng
        self.a = np.zeros(2)
        self.b = math.sqrt(T) * rng.standard_normal(2)
        kk = np.arange(1, tf.JMAX + 1)
        N = 2.0 ** (kk - 1)
        p = 0.5 * 2.0 ** (-gamma * kk) / (2 * N)         # per scalar; 2N scalars per level
        self.t = np.array([tf.threshold_for_tail(pk) for pk in p])  # tau_k for eta and xi_perp
        self.pi = np.array([-math.expm1(2 * n * math.log1p(-pk)) for n, pk in zip(N, p)])
        self.Kstar = tf.sample_last_failure(self.pi, rng)
        self._xi, self._M, self._U = [], [], []
        self._S2 = []          # per level: per-leaf S values (for tests), per level list of arrays by depth
        self._nodes = [np.stack([self.a, self.b])]
        self._cache: dict = {}
        for k in range(1, self.Kstar + 1):
            self._reveal_level(k)

    # ------------------------------------------------------------------
    def _draw_level(self, k: int) -> np.ndarray:  # not used (see _reveal_level)
        raise NotImplementedError

    def _reveal_level(self, k: int) -> None:
        assert k == len(self._xi) + 1
        prev = self._nodes[-1]
        D = prev[1:] - prev[:-1]
        N = len(D)
        delta = self.h / N
        Dn = np.linalg.norm(D, axis=1)
        safe = np.where(Dn > 0, Dn, 1.0)
        s = 0.25 * math.sqrt(delta) * Dn
        nvec = np.stack([D[:, 1], -D[:, 0]], axis=1) / safe[:, None]
        uvec = D / safe[:, None]
        tau = self.t[k - 1]
        if k < self.Kstar:
            eta, xip = self.rng.standard_normal(N), self.rng.standard_normal(N)
        else:
            vec = tf.constrained_vector(self.rng, np.full(2 * N, tau), exceed=(k == self.Kstar))
            eta, xip = vec[:N], vec[N:]
        # tree sigma^2 by depth (leaves depth k-1)
        sig2 = [None] * k
        sig2[k - 1] = s ** 2
        for j in range(k - 2, -1, -1):
            sig2[j] = sig2[j + 1][0::2] + sig2[j + 1][1::2]
        S = np.array([math.sqrt(sig2[0][0]) * eta[0]])
        for j in range(0, k - 1):
            sI2 = sig2[j]
            sL2, sR2 = sig2[j + 1][0::2], sig2[j + 1][1::2]
            sI = np.sqrt(sI2)
            safeI = np.where(sI > 0, sI, 1.0)
            SL = np.where(sI2 > 0, sL2 / np.where(sI2 > 0, sI2, 1.0) * S + np.sqrt(sL2 * sR2) / safeI * eta[2 ** j:2 ** (j + 1)], 0.0)
            SR = S - SL
            S = np.stack([SL, SR], axis=1).reshape(-1)
        e = np.where(s > 0, S / np.where(s > 0, s, 1.0), self.rng.standard_normal(N))
        xi = e[:, None] * nvec + xip[:, None] * uvec
        self._xi.append(xi)
        self._M.append(float(np.max(np.abs(xi))))
        self._U.append(float(max(np.max(np.abs(eta)), 0.0)))
        self._S2.append(0.25 * math.sqrt(delta) * (xi[:, 0] * D[:, 1] - xi[:, 1] * D[:, 0]))  # leaf-level S_c
        new = np.empty((2 * len(prev) - 1, 2))
        new[0::2] = prev
        new[1::2] = 0.5 * (prev[:-1] + prev[1:]) + 0.5 * math.sqrt(delta) * xi
        self._nodes.append(new)

    # ------------------------------------------------------------------ bounds beyond revealed levels
    def _Mbar_unrevealed(self, kk: np.ndarray) -> np.ndarray:
        """|xi_i| <= sqrt(e^2+xip^2) <= tau sqrt(k+1) for unrevealed levels (tail-first thresholds)."""
        return self.t[kk - 1] * np.sqrt(kk + 1.0)

    def Mbar(self, k: int) -> float:
        if k <= len(self._M):
            return self._M[k - 1]
        return float(self.t[k - 1] * math.sqrt(k + 1.0))

    def Ubar(self, k: int) -> float:
        """Certified bound on max |eta| at level k (realised if revealed, threshold otherwise)."""
        return self._U[k - 1] if k <= len(self._U) else float(self.t[k - 1])

    # ------------------------------------------------------------------ Levy-area certificates
    def _A(self, k: int, abar: float, H1: float) -> float:
        return self.Ubar(k) * 0.25 * H1 * self.delta(k) ** abar

    def _sumA(self, n: int, abar: float, H1: float) -> float:
        s = 0.0
        for k in range(n + 1, n + 200):
            term = self._A(k, abar, H1)
            s += term
            if k > self.Kstar + 5 and term < 1e-18 * max(s, 1e-300):
                break
        return s

    def area_dyadic_bound(self, i: int, abar: float, H1: float) -> float:
        """|a_I| <= sqrt(i+1) sqrt|I| sum_{k>i} A_k for dyadic I of depth i."""
        return math.sqrt(i + 1.0) * math.sqrt(self.h * 2.0 ** (-i)) * self._sumA(i, abar, H1)

    def constants(self, alpha: float, abar: float, beta: float, n: int) -> dict:
        """All certified constants of Theorem 4.2 / Lemma 4.1 for the level-n polygon (mesh h_n=T 2^-n)."""
        key = (alpha, abar, beta, n)
        if key in self._cache:
            return self._cache[key]
        T = self.h
        H1 = self.holder_constant(abar)
        # H2: second-level Holder constant (Frobenius norm) via dyadic block decomposition + Chen
        ci = max(self.area_dyadic_bound(i, abar, H1) / (T * 2.0 ** (-i)) ** (2 * abar) for i in range(0, self.Kstar + 60))
        ca = 2.0 / (1.0 - 2.0 ** -abar)
        H2 = 2.0 / (1.0 - 2.0 ** (-2 * abar)) * (0.5 * H1 ** 2 + math.sqrt(2.0) * ci) + 0.5 * H1 ** 2 * ca ** 2
        hn = T * 2.0 ** (-n)
        G = math.sqrt(2.0) * C_B * math.sqrt(n + 1.0) * self._sumA(n, abar, H1) * hn ** (0.5 - 2 * abar)
        theta = hn ** (abar - alpha) if beta <= 2 * alpha else T ** (beta / 2 - alpha) * hn ** (abar - beta / 2)
        cx = 1.0  # Frobenius: |x (x) y| = |x||y|
        first = 4 * H1 * hn ** (abar - alpha)
        short = (3 ** (abar - alpha) * math.sqrt(H2) + math.sqrt(cx) * 3 ** (1 - alpha) * H1) * hn ** (abar - alpha)
        long_ = ((math.sqrt(2) * math.sqrt(H2) + math.sqrt(2 * cx) * H1 + math.sqrt(6 * cx) * H1) * hn ** (abar - alpha)
                 + math.sqrt(G) * theta)
        E = max(first, short, long_)
        Nnorm = max(H1, math.sqrt(H2)) * T ** (abar - alpha) + E
        out = dict(H1=H1, H2=H2, G=G, E=E, Nnorm=Nnorm, h=hn)
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ polygon (canonical lift)
    def cumulative_area(self, n: int) -> np.ndarray:
        P = self.nodes(n)
        dP = P[1:] - P[:-1]
        inc = 0.5 * (P[:-1, 0] * dP[:, 1] - P[:-1, 1] * dP[:, 0])
        return np.concatenate([[0.0], np.cumsum(inc)])

    def residual(self, n: int, m: int, i: int, j: int) -> float:
        """Exact area residual  a^{[m]}_{p,q} - a^{[n]}_{p,q}  over the grid pair of level-n indices (i,j) (m>n):
        equals sum_{k=n+1}^{m} S_k(p,q)  (Chen identity), used to test the generator."""
        r = 2 ** (m - n)
        tot = 0.0
        for k in range(n + 1, m + 1):
            leaf = self._S2[k - 1]
            per = 2 ** (k - 1 - n)     # leaves of level k per level-n grid cell (cell length delta_{n+1}) -> 2^{k-1-n}
            tot += float(np.sum(leaf[i * per:j * per]))
        return tot
