"""Canonical polygonal step-2 lift and grid-based rough-path metrics (m=2).

`lift(nodes, i, j)` returns the canonical signature (increment, second-level matrix) of the polygonal path over
grid indices i<j:  S2 = 1/2 D(x)D + a J,  J=[[0,1],[-1,0]],  a the Levy area (Chen's identity holds exactly).
"""
from __future__ import annotations

import numpy as np

J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])


def cumulative_area(nodes: np.ndarray) -> np.ndarray:
    dP = nodes[1:] - nodes[:-1]
    inc = 0.5 * (nodes[:-1, 0] * dP[:, 1] - nodes[:-1, 1] * dP[:, 0])
    return np.concatenate([[0.0], np.cumsum(inc)])


def area(nodes: np.ndarray, i: int, j: int, cum: np.ndarray | None = None) -> float:
    """Levy area a_{t_i,t_j} of the increments-path (starting at 0)."""
    cum = cumulative_area(nodes) if cum is None else cum
    Pi, Pj = nodes[i], nodes[j]
    return float(cum[j] - cum[i] - 0.5 * (Pi[0] * (Pj[1] - Pi[1]) - Pi[1] * (Pj[0] - Pi[0])))


def lift(nodes: np.ndarray, i: int, j: int, cum: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    D = nodes[j] - nodes[i]
    return D, 0.5 * np.outer(D, D) + area(nodes, i, j, cum) * J2


def chen_defect(nodes: np.ndarray, i: int, k: int, j: int) -> float:
    """‖S_{i,j} - (S_{i,k}+S_{k,j}+D_{i,k}(x)D_{k,j})‖ (should be ~1e-16)."""
    cum = cumulative_area(nodes)
    D1, S1 = lift(nodes, i, k, cum)
    D2, S2 = lift(nodes, k, j, cum)
    D, S = lift(nodes, i, j, cum)
    return float(np.linalg.norm(S - (S1 + S2 + np.outer(D1, D2))))


def rho_alpha_on_grid(nodes_fine: np.ndarray, coarse_level: int, fine_level: int, T: float, alpha: float,
                      stride: int = 1) -> float:
    """Numerical (uncertified!) estimate of rho_alpha between the level-`fine_level` lift (proxy for the true
    Brownian lift) and the level-`coarse_level` polygon lift, over grid pairs of the coarse mesh."""
    r = 2 ** (fine_level - coarse_level)
    Nc = 2 ** coarse_level
    cumF = cumulative_area(nodes_fine)
    nodes_c = nodes_fine[::r]
    cumC = cumulative_area(nodes_c)
    h = T / Nc
    best = 0.0
    idx = list(range(0, Nc + 1, stride))
    for a_ in idx:
        for b_ in idx:
            if b_ <= a_:
                continue
            D, S = lift(nodes_fine, a_ * r, b_ * r, cumF)
            Dc, Sc = lift(nodes_c, a_, b_, cumC)
            L = (b_ - a_) * h
            best = max(best, np.linalg.norm(D - Dc) / L**alpha, np.sqrt(np.linalg.norm(S - Sc)) / L**alpha)
    return float(best)
