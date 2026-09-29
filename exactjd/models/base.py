"""Small interfaces used across the package.

Everything here is a `Protocol`: models only need to *provide* the methods, no inheritance required.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

Array = np.ndarray


@runtime_checkable
class Diffusion(Protocol):
    """Itô diffusion dX = b(t,X)dt + sigma(t,X)dW with d-dim state, m-dim noise."""

    d: int
    m: int

    def drift(self, t: float, x: Array) -> Array: ...
    def sigma(self, t: float, x: Array) -> Array: ...


@runtime_checkable
class TransitionSampler(Protocol):
    """Exact sampler for P_{s,t}(x,.)  (contract D2 of Paper 3)."""

    def sample_transition(self, s: float, x: Array, t: float, rng: np.random.Generator) -> Array: ...


@runtime_checkable
class DirectBridgeSampler(Protocol):
    """Exact bridge values at finitely many interior times.  **Reference only**; the algorithms of
    Papers 2-3 never call it (they only see ε-strong records)."""

    def sample_bridge_values(self, s: float, x: Array, t: float, y: Array, times: Sequence[float],
                             rng: np.random.Generator) -> Array: ...


@runtime_checkable
class BridgeRecordFactory(Protocol):
    """Contract D3: independent refinable bridge records between exact endpoints."""

    def new_bridge(self, s: float, x: Array, t: float, y: Array, rng: np.random.Generator) -> "BridgeRecord": ...


@runtime_checkable
class BridgeRecord(Protocol):
    """A retained latent path with nested certified enclosures.

    `refine(n)` extends the SAME latent path and returns a `PathEnclosure`.
    """

    def refine(self, level: int) -> "PathEnclosure": ...
    def max_norm_interval(self, level: int) -> tuple[float, float]: ...


class PathEnclosure(Protocol):
    """Finite-description continuous path with certified uniform radius: ‖X - center‖∞ ≤ radius."""

    radius: float
    def __call__(self, t: float | Array) -> Array: ...


@runtime_checkable
class JumpMechanism(Protocol):
    """Rate, local bound, mark sampler and update map (Assumption 2.2 of Paper 3)."""

    def intensity(self, t: float, x: Array) -> float: ...
    def local_bound(self, s: float, T: float, radius: float) -> float:
        """Certified upper bound of lambda(t,y) for s<=t<=T, |y|<=radius (may be +0 only if truly zero)."""
    def sample_mark(self, t: float, x: Array, rng: np.random.Generator) -> Any: ...
    def update(self, t: float, x: Array, z: Any) -> Array: ...


class IntensityBoundedGlobally(Protocol):
    """Optional: deterministic global bound (bounded baseline)."""
    global_bound: float
