"""Reproducible independent random streams, lazy uniforms and work accounting.

`StreamFactory` hands out statistically independent `numpy.random.Generator`s keyed by a path of
integers/strings, so every run, segment and trial has its own stream and a run can be resumed by
restoring a stored bit-generator state.  `LazyUniform` is an exact dyadic lazy uniform variable used
for strict retrospective comparisons.  `Budget` implements *explicit* work limits: exceeding one raises
`BudgetExhausted`; nothing is ever silently truncated.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

import numpy as np


def _key(x: int | str) -> int:
    if isinstance(x, int):
        return x
    return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8], "little")


class StreamFactory:
    """Independent streams via `SeedSequence` spawn keys."""

    def __init__(self, seed: int = 0, path: tuple = ()):
        self.seed = seed
        self.path = tuple(path)

    def child(self, *names: int | str) -> "StreamFactory":
        return StreamFactory(self.seed, self.path + names)

    def generator(self, *names: int | str) -> np.random.Generator:
        ss = np.random.SeedSequence(entropy=self.seed, spawn_key=tuple(_key(n) for n in self.path + names))
        return np.random.Generator(np.random.PCG64(ss))


def get_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def set_state(rng: np.random.Generator, state: dict) -> None:
    rng.bit_generator.state = state


class BudgetExhausted(RuntimeError):
    """Raised when an explicit work budget is exceeded (run must be recorded as censored)."""


@dataclass
class Budget:
    """Work counters + optional limits.  `tick` is called by primitives."""

    max_calls: int | None = None
    max_seconds: float | None = None
    calls: dict[str, int] = field(default_factory=dict)
    t0: float = field(default_factory=time.perf_counter)

    def tick(self, name: str, n: int = 1) -> None:
        self.calls[name] = self.calls.get(name, 0) + n
        if self.max_calls is not None and self.total() > self.max_calls:
            raise BudgetExhausted(f"call budget {self.max_calls} exceeded")
        if self.max_seconds is not None and (self.calls[name] & 63) == 0:
            if time.perf_counter() - self.t0 > self.max_seconds:
                raise BudgetExhausted(f"time budget {self.max_seconds}s exceeded")

    def total(self) -> int:
        return sum(self.calls.values())

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


NULL_BUDGET = Budget()


class LazyUniform:
    """Exact lazy Unif(0,1): dyadic interval [k/2^n,(k+1)/2^n] shrinking by fair bits."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.n = 0
        self.k = 0

    def refine(self, bits: int = 1) -> None:
        b = int(self.rng.integers(0, 1 << bits))
        self.k = (self.k << bits) | b
        self.n += bits

    def interval(self) -> tuple[Fraction, Fraction]:
        d = 1 << self.n
        return Fraction(self.k, d), Fraction(self.k + 1, d)

    def less_than(self, enclosure: Callable[[int], tuple[float, float]], budget: Budget = NULL_BUDGET) -> bool:
        """Decide `U < Q` where `enclosure(j)` returns an interval containing Q that shrinks with j.

        Refines both sides until the strict comparison is certified (a.s. finite).
        """
        j = 0
        while True:
            budget.tick("lazy_compare")
            qlo, qhi = enclosure(j)
            lo, hi = self.interval()
            if hi <= Fraction(qlo):
                return True
            if lo >= Fraction(qhi):
                return False
            self.refine(2)
            j += 1

    def compare_value(self, x: float) -> bool:
        """`U < x` for an exactly known real."""
        fx = Fraction(x)
        while True:
            lo, hi = self.interval()
            if hi <= fx:
                return True
            if lo >= fx:
                return False
            self.refine(4)

    def value_interval_float(self) -> tuple[float, float]:
        lo, hi = self.interval()
        return float(lo), float(hi)
