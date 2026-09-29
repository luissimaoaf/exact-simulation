"""Bernoulli factories used by the countable race (Paper 2 §3.3).

* `thinning_factory`  : C<=1, one p-coin then an independent rational coin  (exact, paper).
* `linear_factory`    : C>1, Cp<=1-eta.  **Not Huber's (2016) algorithm** (whose pseudo-code is not reproduced in
  the paper); an independent construction with a short exactness proof, same contract as Paper 2 needs:
      - uses only the coin bits and independent auxiliary randomness,
      - terminates surely once its finitely many coins are flipped,
      - output 1 requires at least one head (Lemma 3.5 'head property'),
      - expected coins O(C/eta^2) for the finite-mean stage (plus a heavy-tailed rare stage).
  Construction.  Let x1=(1-eta)/C, w=eta/C and f^ concave, nondecreasing, C^1:
        f^(x)=Cx (x<=x1),  Cx1+C(x-x1)-C(x-x1)^2/(2w) (x1<=x<=x1+w),  1-eta/2 (x>=x1+w).
  f^=Cx on the admissible range p<=x1 and f^''>=-kappa, kappa=C^2/eta.  Bernstein coefficients a(n,k)=f^(k/n)
  satisfy  E[a(n,S_n)|S_{n+1}=s]<=a(n+1,s)  (Jensen), with increments
        d(n+1,s)=a(n+1,s)-[(s/(n+1))a(n,s-1)+(1-s/(n+1))a(n,s)]  in [0, kappa Var/2],
  Var<=x̄(1-x̄)/n^2.  Where d>0 the hull contains [x1,x1+w] so x̄<=1/C+1/n and for n>=C+1
        d(n+1,s) <= C/(eta n^2).
  Since E[d(n,S_n)]=B_n f^ - B_{n-1} f^ the series B_{n0} f^ + sum_{n>n0} E d(n,S_n) telescopes to f^(p)=Cp.
  Randomised truncation: N=n0 w.p. 1-eta/2 (weight a/(1-eta/2)<=1 since a<=1-eta... uses a<=1-eta/2), N=n>n0
  w.p. pi_n=(eta/2)(n-1)^{-2}/psi1(n0) (>= bound on d, because psi1(n0)<=1/(n0-1)<=eta^2/(2C)); output Bernoulli(weight).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import polygamma

from ..rng import NULL_BUDGET, Budget

Coin = Callable[[], int]


def thinning_factory(C: float, coin: Coin, rng: np.random.Generator) -> tuple[int, int]:
    """Bernoulli(Cp) for 0<C<=1: one p-coin, then independent Bernoulli(C).  Returns (out, coins)."""
    assert 0 < C <= 1
    b = coin()
    return (int(b and rng.random() < C), 1)


def _fhat(x: np.ndarray, C: float, eta: float) -> np.ndarray:
    x1, w = (1 - eta) / C, eta / C
    x = np.asarray(x, float)
    mid = C * x1 + C * (x - x1) - C * (x - x1) ** 2 / (2 * w)
    return np.where(x <= x1, C * x, np.where(x <= x1 + w, mid, 1 - eta / 2))


def linear_n0(C: float, eta: float) -> int:
    return int(math.ceil(1 + max(C, 2 * C / eta**2)))


def _sample_tail_N(n0: int, rng: np.random.Generator) -> int:
    """N>n0 with P(N=n) ∝ (n-1)^-2  (P(N>=n)=psi1(n-1)/psi1(n0))."""
    u = 1.0 - rng.random()
    tot = float(polygamma(1, n0))
    lo, hi = n0 + 1, n0 + 2
    while float(polygamma(1, hi - 1)) / tot > u:       # G(n) = P(N>=n) decreasing
        lo, hi = hi, hi * 2
        if hi > 2**62:
            return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(polygamma(1, mid - 1)) / tot > u:
            lo = mid
        else:
            hi = mid
    return lo


def linear_factory(C: float, eta: float, coin: Coin, rng: np.random.Generator,
                   budget: Budget = NULL_BUDGET) -> tuple[int, int]:
    """Bernoulli(Cp) for C>1 and Cp<=1-eta (eta in (0,1)).  Returns (out, coins used)."""
    assert C > 1 and 0 < eta < 1
    n0 = linear_n0(C, eta)
    tot = float(polygamma(1, n0))
    if rng.random() < 1 - eta / 2:
        N = n0
    else:
        N = _sample_tail_N(n0, rng)
    if N > n0 + 10**12:
        budget.tick("factory_coin", 10**18)
    S = 0
    for _ in range(N):
        budget.tick("factory_coin")
        S += coin()
    if S == 0:
        return 0, N
    if N == n0:
        a = float(_fhat(S / N, C, eta))
        val = a / (1 - eta / 2)
    else:
        m = N
        lam = S / m
        d = float(_fhat(S / m, C, eta)
                  - (lam * _fhat((S - 1) / (m - 1), C, eta) + (1 - lam) * _fhat(S / (m - 1), C, eta)))
        pi = (eta / 2) * (m - 1.0) ** -2 / tot
        val = d / pi
    val = min(max(val, 0.0), 1.0 + 1e-12)
    return int(rng.random() < val), N


def bernoulli_factory(C: float, coin: Coin, rng: np.random.Generator, eta: float = 0.5,
                      budget: Budget = NULL_BUDGET) -> tuple[int, int]:
    """Dispatch: thinning for C<=1, linear factory otherwise (Cp<=1-eta must hold)."""
    if C <= 1:
        return thinning_factory(C, coin, rng)
    return linear_factory(C, eta, coin, rng, budget)
