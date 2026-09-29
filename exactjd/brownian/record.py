"""Retained nested Brownian (bridge) record with certified uniform error.

Levy-Ciesielski (midpoint) construction on [0,h]:
  level 0: chord from a to b (b drawn N(a, h I) for free Brownian motion, given for a bridge);
  level k>=1: 2^{k-1} cells of length delta_k = h 2^{1-k}; midpoint = chord midpoint + (sqrt(delta_k)/2) xi,
              xi iid N(0,I_d).

Certificate (`radius(n)`): ‖W - polygon_n‖∞ <= sum_{k>n} sqrt(d) (sqrt(delta_k)/2) Mbar_k, where Mbar_k is the
realised max |coordinate| for revealed levels and the deterministic tail-first threshold t_k beyond K*.
Refinement never changes previously revealed values (nested, same latent path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import tailfirst as tf


def default_tail_probs(kmax: int, gamma: float, counts: np.ndarray) -> np.ndarray:
    """Per-scalar tail probabilities so that P(F_k) ~ 2^{-gamma k} (cost E[2^{K*}] finite for gamma>1)."""
    k = np.arange(1, kmax + 1)
    pi_target = 0.5 * 2.0 ** (-gamma * k)
    return pi_target / counts


class BrownianRecord:
    """d-dimensional Brownian path on [0,h] (bridge if `b` given, free BM otherwise)."""

    GAMMA = 2.5

    def __init__(self, h: float, d: int, rng: np.random.Generator, a=None, b=None, gamma: float | None = None):
        self.h, self.d, self.rng = float(h), int(d), rng
        gamma = self.GAMMA if gamma is None else gamma
        self.a = np.zeros(d) if a is None else np.asarray(a, float).copy()
        self.b = (self.a + math.sqrt(h) * rng.standard_normal(d)) if b is None else np.asarray(b, float).copy()
        # thresholds t_k (per scalar coordinate) and failure probs pi_k for levels 1..JMAX
        kk = np.arange(1, tf.JMAX + 1)
        counts = d * 2.0 ** (kk - 1)
        p = default_tail_probs(tf.JMAX, gamma, counts)
        self.t = np.array([tf.threshold_for_tail(pk) for pk in p])
        self.pi = np.array([-math.expm1(c * math.log1p(-pk)) for c, pk in zip(counts, p)])
        self.Kstar = tf.sample_last_failure(self.pi, rng)
        self._xi: list[np.ndarray] = []     # xi[k-1] shape (2^{k-1}, d)
        self._nodes: list[np.ndarray] = [np.stack([self.a, self.b])]
        self._M: list[float] = []           # realised max |coord| per revealed level
        for k in range(1, self.Kstar + 1):
            self._reveal_level(k)

    # ------------------------------------------------------------------ level generation
    def _draw_level(self, k: int) -> np.ndarray:
        n = 2 ** (k - 1)
        if k < self.Kstar:
            return self.rng.standard_normal((n, self.d))
        tvec = np.full(n * self.d, self.t[k - 1])
        return tf.constrained_vector(self.rng, tvec, exceed=(k == self.Kstar)).reshape(n, self.d)

    def _reveal_level(self, k: int) -> None:
        assert k == len(self._xi) + 1
        xi = self._draw_level(k)
        self._xi.append(xi)
        self._M.append(float(np.max(np.abs(xi))))
        prev = self._nodes[-1]
        delta = self.h * 2.0 ** (1 - k)
        new = np.empty((2 * len(prev) - 1, self.d))
        new[0::2] = prev
        new[1::2] = 0.5 * (prev[:-1] + prev[1:]) + 0.5 * math.sqrt(delta) * xi
        self._nodes.append(new)

    def ensure(self, n: int) -> None:
        while len(self._xi) < n:
            self._reveal_level(len(self._xi) + 1)

    # ------------------------------------------------------------------ certified quantities
    def Mbar(self, k: int) -> float:
        """Certified bound on max |xi coordinate| at level k: realised maximum if the level is revealed
        (any revealed level, also beyond K*), deterministic tail-first threshold otherwise."""
        if k <= len(self._M):
            return self._M[k - 1]
        return float(self.t[k - 1])

    def delta(self, k: int) -> float:
        return self.h * 2.0 ** (1 - k)

    def _Mvec(self) -> np.ndarray:
        key = len(self._M)
        c = getattr(self, "_mv_cache", None)
        if c is None or c[0] != key:
            kk = np.arange(1, tf.JMAX + 1)
            mv = np.array(self.t[:tf.JMAX], float)
            mv[:key] = self._M[:key]
            if hasattr(self, "_Mbar_unrevealed"):
                mv[key:] = self._Mbar_unrevealed(kk[key:])
            self._mv_cache = (key, mv)
            c = self._mv_cache
        return c[1]

    def nodes(self, n: int) -> np.ndarray:
        self.ensure(n)
        return self._nodes[n]

    def radius(self, n: int) -> float:
        """Certified ‖W - polygon_n‖∞ (Euclidean norm): sum_{k>n} sqrt(d) (sqrt(delta_k)/2) Mbar_k."""
        n = max(n, 0)
        kk = np.arange(1, tf.JMAX + 1)
        terms = math.sqrt(self.d) * 0.5 * np.sqrt(self.h * 2.0 ** (1.0 - kk)) * self._Mvec()
        return float(terms[n:].sum())

    def polygon(self, n: int):
        nodes = self.nodes(n)
        grid = np.linspace(0.0, self.h, len(nodes))

        def ev(t):
            t = np.atleast_1d(np.asarray(t, float))
            return np.stack([np.interp(t, grid, nodes[:, j]) for j in range(self.d)], axis=-1)
        return ev

    def increment_bound(self, l: float) -> float:
        """Certified sup over intervals of length l of |W_t - W_s| (Euclidean)."""
        kk = np.arange(1, tf.JMAX + 1)
        dk = self.h * 2.0 ** (1.0 - kk)
        tot = np.linalg.norm(self.b - self.a) * min(l / self.h, 1.0)
        return float(tot + math.sqrt(self.d) * np.sum(self._Mvec() * np.minimum(l / np.sqrt(dk), np.sqrt(dk))))

    def holder_constant(self, alpha: float) -> float:
        """Certified H with |W_t-W_s| <= H |t-s|^alpha for all s,t in [0,h]  (concave-piecewise-linear increments:
        sup at the breakpoints l=h 2^-j, j=0..300)."""
        best = 0.0
        for j in range(0, 300):
            l = self.h * 2.0 ** (-j)
            best = max(best, self.increment_bound(l) / l**alpha)
        return best

    def endpoint_modulus(self, s: float, n: int, at_end: bool = True) -> float:
        """Certified bound on |W_t - polygon_n(t)| at distance s from the END of a *bridge* (both vanish at T):
        sqrt(d) sum_{k>n} Mbar_k min(s/sqrt(delta_k), sqrt(delta_k)/2)."""
        kk = np.arange(n + 1, tf.JMAX + 1)
        dk = self.h * 2.0 ** (1.0 - kk)
        return float(math.sqrt(self.d) * np.sum(self._Mvec()[n:] * np.minimum(s / np.sqrt(dk), 0.5 * np.sqrt(dk))))
