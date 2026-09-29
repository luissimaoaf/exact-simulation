"""Jump mechanisms (rate, local bound, marks, updates) for the benchmark models."""
from __future__ import annotations

import numpy as np

from .base import Array
from .nonlinear import Phi, Psi


class _CoordMarks:
    """Independent symmetric coordinate marks Z_j in {-jump,+jump} (Paper 3 E.2)."""

    def __init__(self, d: int, jump: float = 0.2):
        self.d, self.jump = d, jump

    def sample_mark(self, t: float, x: Array, rng: np.random.Generator) -> Array:
        return self.jump * (2.0 * rng.integers(0, 2, self.d) - 1.0)

    @property
    def m2(self) -> float:
        return self.d * self.jump**2


class ConstantRateJump(_CoordMarks):
    """lambda == lam, additive update x+Z (benchmark A).  Globally bounded."""

    def __init__(self, d: int, lam: float = 2.0, jump: float = 0.2):
        super().__init__(d, jump)
        self.lam = lam
        self.global_bound = lam

    def intensity(self, t, x):
        return self.lam

    def local_bound(self, s, T, radius):
        return self.lam

    def update(self, t, x, z):
        return x + z


class BoundedStateRateJump(_CoordMarks):
    """lambda = 1 + 3|x|^2/(1+|x|^2) <= 4 (benchmark B), contractive update rho x + Z."""

    def __init__(self, d: int, rho: float = 0.5, jump: float = 0.2):
        super().__init__(d, jump)
        self.rho = rho
        self.global_bound = 4.0

    def intensity(self, t, x):
        r2 = float(np.dot(x, x))
        return 1.0 + 3.0 * r2 / (1.0 + r2)

    def local_bound(self, s, T, radius):
        r2 = radius**2
        return 1.0 + 3.0 * r2 / (1.0 + r2)

    def update(self, t, x, z):
        return self.rho * x + z


class QuadraticRateJump(_CoordMarks):
    """lambda = lam0 + lam1|x|^2 (globally unbounded), Gamma = rho x + Z  (Paper 3 §6.1)."""

    def __init__(self, d: int, lam0: float = 1.0, lam1: float = 1.0, rho: float = 0.5, jump: float = 0.2):
        super().__init__(d, jump)
        self.lam0, self.lam1, self.rho = lam0, lam1, rho

    def intensity(self, t, x):
        return self.lam0 + self.lam1 * float(np.dot(x, x))

    def local_bound(self, s, T, radius):
        return self.lam0 + self.lam1 * radius**2

    def update(self, t, x, z):
        return self.rho * x + z


class TransformedQuadraticRateJump(_CoordMarks):
    """Nonlinear example: lambda = lam0 + lam1|Psi(x)|^2, Gamma = Phi(rho Psi(x)+Z); |Psi(x)|<=|x|+1/2."""

    def __init__(self, lam0: float = 1.0, lam1: float = 1.0, rho: float = 0.5, jump: float = 0.2):
        super().__init__(2, jump)
        self.lam0, self.lam1, self.rho = lam0, lam1, rho

    def intensity(self, t, x):
        p = Psi(np.asarray(x, float))
        return self.lam0 + self.lam1 * float(np.dot(p, p))

    def local_bound(self, s, T, radius):
        return self.lam0 + self.lam1 * (radius + 0.5) ** 2

    def update(self, t, x, z):
        return Phi(self.rho * Psi(np.asarray(x, float)) + z)
