"""The papers' nonlinear 2-d benchmark: nonconstant covariance, noncommuting noise, law = image of BM.

  f(z)=z+sin(z)/2,  g=f^{-1},  q(x1)=1+cos(g(x1))/2,
  b(x)=(-sin(g(x1))/4, 0),  sigma(x)=[[q cos x2, -q sin x2],[sin x2, cos x2]],  a=diag(q^2,1).

Law of the diffusion == law of Phi(B), Phi(z)=(f(z1),z2), B 2-d Brownian motion started at Psi(x)=Phi^{-1}(x)
(Paper 1 Prop 10.1 / Paper 3 App. D).  The *direct transformed reference* below uses that identity and is NOT
an implementation of Papers 1-2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import Array


def f(z):
    return z + 0.5 * np.sin(z)


def fprime(z):
    return 1.0 + 0.5 * np.cos(z)


def g(x, tol: float = 1e-14):
    """Inverse of f: monotone, |g(x)-x|<=1/2; safeguarded Newton inside the bracket [x-1/2,x+1/2]."""
    x = np.asarray(x, float)
    lo, hi = x - 0.5, x + 0.5
    z = x.copy()
    for _ in range(100):
        r = f(z) - x
        lo = np.where(r < 0, z, lo)
        hi = np.where(r > 0, z, hi)
        zn = z - r / fprime(z)
        bad = (zn <= lo) | (zn >= hi)
        zn = np.where(bad, 0.5 * (lo + hi), zn)
        if np.max(np.abs(zn - z)) < tol:
            z = zn
            break
        z = zn
    return z


def g_scalar(x: float) -> float:
    """Scalar Newton inverse of f (fast path used inside ODE loops)."""
    z = x
    for _ in range(60):
        dz = (z + 0.5 * math.sin(z) - x) / (1.0 + 0.5 * math.cos(z))
        z -= dz
        if abs(dz) < 1e-15:
            break
    return z


def Phi(z: Array) -> Array:
    z = np.asarray(z, float)
    return np.stack([f(z[..., 0]), z[..., 1]], axis=-1)


def Psi(x: Array) -> Array:
    x = np.asarray(x, float)
    return np.stack([g(x[..., 0]), x[..., 1]], axis=-1)


def q(x1):
    return fprime(g(x1))


class NonlinearBenchmark:
    """Coefficients + exact (transformed Gaussian) transitions and direct bridge reference."""

    d = 2
    m = 2
    # Safe constants of Paper 2 Cor. 9.3
    Lb, B0, Sigma0, Sigmax, Sigmat = 0.5, 0.25, 2.0, 3.0, 0.0

    @staticmethod
    def heat_kernel_constants(T: float) -> tuple[float, float]:
        """(C+, kappa+) of (S3): C+ = e^{T/72}/pi, kappa+ = 1/9."""
        return float(np.exp(T / 72.0) / np.pi), 1.0 / 9.0

    def drift(self, t: float, x: Array) -> Array:
        return np.array([-0.25 * np.sin(g(x[0])), 0.0])

    def sigma(self, t: float, x: Array) -> Array:
        qq, c, s = q(x[0]), np.cos(x[1]), np.sin(x[1])
        return np.array([[qq * c, -qq * s], [s, c]])

    def a(self, x: Array) -> Array:
        return np.diag([q(x[0]) ** 2, 1.0])

    def F(self, x: Array) -> Array:
        """Regular Stratonovich drift b - 1/2 sum_j D sigma_j sigma_j = (q/2, 0)  (Paper 1 §10)."""
        return np.array([0.5 * q(x[0]), 0.0])

    # exact transition: X_{s+h} = Phi(Psi(x) + sqrt(h) G)
    def sample_transition(self, s: float, x: Array, t: float, rng: np.random.Generator) -> Array:
        return Phi(Psi(np.asarray(x, float)) + np.sqrt(t - s) * rng.standard_normal(2))

    def transition_density(self, h: float, x: Array, y: Array) -> float:
        """p_h(x,y) = exp{-[(g(y1)-g(x1))^2+(y2-x2)^2]/(2h)} / (2 pi h q(y1))  (Paper 2 Cor 9.3)."""
        gy, gx = g(y[0]), g(x[0])
        return float(np.exp(-((gy - gx) ** 2 + (y[1] - x[1]) ** 2) / (2 * h)) / (2 * np.pi * h * fprime(gy)))

    # DIRECT TRANSFORMED REFERENCE bridge values (not Papers 1-2)
    def sample_bridge_values(self, s: float, x: Array, t: float, y: Array, times: Sequence[float],
                             rng: np.random.Generator) -> Array:
        zx, zy = Psi(np.asarray(x, float)), Psi(np.asarray(y, float))
        T = t - s
        out = []
        cur_t, cur = s, zx
        for ti in times:
            r = t - cur_t
            mean = cur + (ti - cur_t) / r * (zy - cur)
            var = (ti - cur_t) * (t - ti) / r
            cur = mean + np.sqrt(var) * rng.standard_normal(2)
            cur_t = ti
            out.append(Phi(cur))
        return np.array(out)
