"""BASELINES: conventional time-discretised simulators.  APPROXIMATE - never labelled exact.

Used only as comparison references (grid sensitivity must be reported, Paper 3 App. E.2).
"""
from __future__ import annotations

import numpy as np

from ..models.base import Array


def euler_maruyama(model, x0: Array, T: float, n_steps: int, rng: np.random.Generator,
                   dW: Array | None = None) -> tuple[Array, Array]:
    """Euler-Maruyama path on a uniform grid.  `dW` (n_steps, m) allows common Brownian increments."""
    h = T / n_steps
    x = np.asarray(x0, float).copy()
    path = np.empty((n_steps + 1, model.d))
    path[0] = x
    if dW is None:
        dW = np.sqrt(h) * rng.standard_normal((n_steps, model.m))
    for k in range(n_steps):
        t = k * h
        x = x + model.drift(t, x) * h + model.sigma(t, x) @ dW[k]
        path[k + 1] = x
    return np.linspace(0, T, n_steps + 1), path


def euler_jump_diffusion(model, jump, x0: Array, T: float, n_steps: int, rng: np.random.Generator,
                         dW: Array | None = None):
    """Euler scheme with jump probability 1-exp(-lambda(x) h) per step (explicitly approximate hazard).

    Returns (events, x_T) with events = list of (time, pre, mark, post).
    """
    h = T / n_steps
    x = np.asarray(x0, float).copy()
    events = []
    if dW is None:
        dW = np.sqrt(h) * rng.standard_normal((n_steps, model.m))
    u = rng.random(n_steps)
    for k in range(n_steps):
        t = k * h
        x = x + model.drift(t, x) * h + model.sigma(t, x) @ dW[k]
        lam = jump.intensity(t + h, x)
        if u[k] < 1.0 - np.exp(-lam * h):
            z = jump.sample_mark(t + h, x, rng)
            xn = jump.update(t + h, x, z)
            events.append((t + h, x.copy(), z, xn.copy()))
            x = xn
    return events, x
