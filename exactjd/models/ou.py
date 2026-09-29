"""Multivariate Ornstein-Uhlenbeck / linear Gaussian diffusion  dZ = (B Z + c)dt + Sigma dW.

Provides exact transitions, exact Gaussian bridge values (reference only), analytic moments and the
first-event survival formula of Paper 3, App. E.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm

from .base import Array


@dataclass
class OU:
    B: Array
    c: Array
    Sigma: Array

    def __post_init__(self):
        self.B = np.atleast_2d(np.asarray(self.B, float))
        self.d = self.B.shape[0]
        self.c = np.broadcast_to(np.asarray(self.c, float), (self.d,)).copy()
        self.Sigma = np.atleast_2d(np.asarray(self.Sigma, float))
        self.m = self.Sigma.shape[1]
        self.a = self.Sigma @ self.Sigma.T

    @classmethod
    def isotropic(cls, kappa: float, Sigma: Array, c=0.0) -> "OU":
        Sigma = np.atleast_2d(np.asarray(Sigma, float))
        return IsoOU(-kappa * np.eye(Sigma.shape[0]), c, Sigma)

    # --- Diffusion protocol
    def drift(self, t: float, x: Array) -> Array:
        return self.B @ x + self.c

    def sigma(self, t: float, x: Array) -> Array:
        return self.Sigma

    # --- exact Gaussian transition
    def transition_params(self, h: float) -> tuple[Array, Array, Array]:
        """(A, m0, Q) with Z_{s+h}|x ~ N(A x + m0, Q)  (Van Loan for Q, block-matrix for m0)."""
        d = self.d
        M = np.zeros((2 * d, 2 * d))
        M[:d, :d] = -self.B
        M[:d, d:] = self.a
        M[d:, d:] = self.B.T
        E = expm(M * h)
        A = E[d:, d:].T
        Q = A @ E[:d, d:]
        Q = 0.5 * (Q + Q.T)
        N = np.zeros((d + 1, d + 1))
        N[:d, :d] = self.B
        N[:d, d] = self.c
        m0 = expm(N * h)[:d, d]
        return A, m0, Q

    def mean_cov(self, h: float, x: Array) -> tuple[Array, Array]:
        A, m0, Q = self.transition_params(h)
        return A @ x + m0, Q

    def sample_transition(self, s: float, x: Array, t: float, rng: np.random.Generator) -> Array:
        mu, Q = self.mean_cov(t - s, np.asarray(x, float))
        return mu + np.linalg.cholesky(Q + 1e-300 * np.eye(self.d)) @ rng.standard_normal(self.d)

    def sample_path_values(self, x0: Array, times: Sequence[float], rng: np.random.Generator) -> Array:
        out, x, t0 = [], np.asarray(x0, float), 0.0
        for t in times:
            x = self.sample_transition(t0, x, t, rng)
            out.append(x)
            t0 = t
        return np.array(out)

    # --- exact Gaussian bridge values at interior times (REFERENCE ONLY)
    def sample_bridge_values(self, s: float, x: Array, t: float, y: Array, times: Sequence[float],
                             rng: np.random.Generator) -> Array:
        times = list(times)
        d = self.d
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        res, cur_t, cur = [], s, x
        # sequential conditional sampling: X_{t_i} | X_{t_{i-1}}, X_T=y (Markov)
        for ti in times:
            A1, m1, Q1 = self.transition_params(ti - cur_t)
            A2, m2, Q2 = self.transition_params(t - ti)
            # joint: X_ti ~ N(mu1,Q1); y | X_ti ~ N(A2 X_ti + m2, Q2)
            mu1 = A1 @ cur + m1
            S = A2 @ Q1 @ A2.T + Q2
            K = Q1 @ A2.T @ np.linalg.inv(S)
            mu = mu1 + K @ (y - A2 @ mu1 - m2)
            C = Q1 - K @ A2 @ Q1
            C = 0.5 * (C + C.T)
            w, V = np.linalg.eigh(C)
            L = V * np.sqrt(np.clip(w, 0, None))
            cur = mu + L @ rng.standard_normal(d)
            cur_t = ti
            res.append(cur)
        return np.array(res)

    # --- analytic first-event survival for OU with lambda(y)=lam0+lam1|y|^2, B=-kappa I, diagonal a
    def survival_quadratic_rate(self, t: float, x: Array, lam0: float, lam1: float) -> float:
        """P_x(tau_1>t) = exp{-lam0 t - c(t) - x^T A(t) x}  (Paper 3, App. E.1, diagonal a; kappa from B=-kappa I)."""
        kappa = -self.B[0, 0]
        assert np.allclose(self.B, -kappa * np.eye(self.d)), "closed form needs B=-kappa*I"
        # rotate a to diagonal: quadratic rate is rotation invariant
        w, V = np.linalg.eigh(self.a)
        xr = V.T @ np.asarray(x, float)
        beta = np.sqrt(kappa**2 + 2 * w * lam1)
        A = lam1 * np.sinh(beta * t) / (beta * np.cosh(beta * t) + kappa * np.sinh(beta * t))
        cterm = 0.5 * np.log(np.cosh(beta * t) + kappa / beta * np.sinh(beta * t)) - 0.5 * kappa * t
        return float(np.exp(-lam0 * t - cterm.sum() - np.sum(A * xr**2)))


class IsoOU(OU):
    """B=-kappa I, c=0: closed-form transitions (mean e^{-kh}x, cov (1-e^{-2kh})/(2k) a)."""

    def __post_init__(self):
        super().__post_init__()
        self.kappa = -float(self.B[0, 0])
        self._La = np.linalg.cholesky(self.a)

    def sample_transition(self, s, x, t, rng):
        h = t - s
        k = self.kappa
        var = h if k == 0 else -np.expm1(-2 * k * h) / (2 * k)
        return np.exp(-k * h) * np.asarray(x, float) + np.sqrt(var) * (self._La @ rng.standard_normal(self.d))
