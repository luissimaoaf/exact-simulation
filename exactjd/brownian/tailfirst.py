"""Exact "tail-first" sampling utilities.

Certificates for the *unrevealed* part of a Brownian hierarchy must hold surely on the event of
probability one that we simulate.  The device used here (the same principle as BCD's random constants,
realised with independent per-level events of exactly known probability):

  * level k has an event  F_k = {some constrained iid N(0,1) coordinate exceeds its threshold},
    P(F_k)=pi_k computed exactly, F_k independent across levels;
  * K* = last k with F_k (0 if none) has  P(K* <= k) = prod_{j>k}(1-pi_j)  and is sampled exactly;
  * given K*=k*: levels j<k* are unconditioned, level k* is conditioned on F, levels j>k* on not F
    (iid truncated normals).  The joint law is exactly the unconditioned one, and for levels >K*
    the thresholds are *deterministic* bounds.

Arithmetic caveat: probabilities/quantiles are evaluated in double precision (ideal-arithmetic proof,
float implementation).  Tails are truncated at level `JMAX` where pi_j < 1e-30.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfcinv, ndtr, ndtri

JMAX = 400


def threshold_for_tail(p: float) -> float:
    """t with P(|Z|>t)=p."""
    return float(math.sqrt(2.0) * erfcinv(min(max(p, 1e-300), 1.0)))


def tail_prob(t: np.ndarray | float):
    """P(|Z|>t)."""
    return 2.0 * ndtr(-np.asarray(t, float))


def failure_probability(t: np.ndarray) -> float:
    """1 - prod_i (1 - p_i), p_i = P(|Z|>t_i)  (exact, stable)."""
    p = tail_prob(t)
    return float(-np.expm1(np.sum(np.log1p(-np.minimum(p, 1 - 1e-16)))))


def truncated_normal(rng: np.random.Generator, t, size=None) -> np.ndarray:
    """iid N(0,1) conditioned on |z|<=t (t scalar or array broadcastable to size)."""
    lo = ndtr(-np.asarray(t, float))
    u = rng.random(size)
    z = ndtri(lo + u * (1.0 - 2.0 * lo))
    return np.clip(z, -t, t)


def tail_normal(rng: np.random.Generator, t: float) -> float:
    """N(0,1) conditioned on |z|>t."""
    u = 1.0 - rng.random()  # (0,1]
    z = -ndtri(u * ndtr(-t))
    return float(z if rng.random() < 0.5 else -z)


def constrained_vector(rng: np.random.Generator, t: np.ndarray, exceed: bool) -> np.ndarray:
    """iid N(0,1) vector.  exceed=False: conditioned on |z_i|<=t_i for all i.
    exceed=True: conditioned on max_i(|z_i|-t_i)>0, sampled exactly by locating the first exceedance."""
    t = np.asarray(t, float)
    n = t.size
    if not exceed:
        return truncated_normal(rng, t, n)
    p = np.minimum(tail_prob(t), 1 - 1e-16)
    logsurv = np.concatenate([[0.0], np.cumsum(np.log1p(-p))[:-1]])
    logw = logsurv + np.log(p)
    w = np.exp(logw - logw.max())
    i = int(rng.choice(n, p=w / w.sum()))
    z = np.empty(n)
    if i > 0:
        z[:i] = truncated_normal(rng, t[:i], i)
    z[i] = tail_normal(rng, float(t[i]))
    if i + 1 < n:
        z[i + 1:] = rng.standard_normal(n - i - 1)
    return z


def sample_last_failure(pi: np.ndarray, rng: np.random.Generator) -> int:
    """K* = last failing level, levels indexed 1..len(pi); returns 0 if no failure.
    `pi[j-1]` = P(F_j).  P(K*<=k) = prod_{j>k}(1-pi_j).  Exact inversion with a uniform."""
    logs = np.log1p(-np.minimum(pi, 1 - 1e-16))
    # tail sums: logG[k] = sum_{j>k} log(1-pi_j), k=0..len
    logG = np.concatenate([np.cumsum(logs[::-1])[::-1], [0.0]])  # logG[k]=sum_{j>=k+1}
    u = rng.random()
    lu = math.log(max(u, 1e-300))
    k = int(np.searchsorted(logG, lu, side="left"))  # logG increasing in k
    # logG[k] >= lu first index
    while k < len(logG) - 1 and logG[k] < lu:
        k += 1
    return k
