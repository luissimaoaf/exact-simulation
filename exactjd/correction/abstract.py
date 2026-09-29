"""Paper 2 §3: exact retrospective acceptance-rejection for an unbounded positive weight.

Abstract objects
  LogWeightOracle : `new_proposal(rng)` -> ProposalRecord with nested enclosures `interval(n)` of J(X)=log Psi(X)
                    (`interval` may be called with increasing n; each call refines the SAME latent object).
  Envelope        : summable deterministic majorant d_k of V_k q_k(theta); `LogEnvelope(Mc)` = eq. (7),
                    `PowerEnvelope(M, eps)` = Prop 4.9.
Algorithm 3.6 (`retrospective_select`): shared shift Theta, countable race with membership coins (Alg 3.3) feeding a
Bernoulli factory (thinning C<=1, linear factory C>1), retained first-head proposal (Lemma 3.5), final rejection.

Nothing here reads a likelihood value except through strict retrospective comparisons; the factory sees only bits.
Work is counted; exceeding an explicit `Budget` raises `BudgetExhausted` (never silently truncated).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..rng import NULL_BUDGET, Budget, BudgetExhausted, LazyUniform
from .factory import bernoulli_factory, linear_n0

LN2 = math.log(2.0)


class ProposalRecord(Protocol):
    def interval(self, n: int) -> tuple[float, float]: ...


class LogWeightOracle(Protocol):
    def new_proposal(self, rng: np.random.Generator) -> ProposalRecord: ...


# --------------------------------------------------------------------------- envelopes
class Envelope(Protocol):
    D: float
    def sample_k(self, rng: np.random.Generator) -> int: ...
    def log_C(self, k: int) -> float: ...       # log of C_k = V_k/(2 d_k)


class LogEnvelope:
    """d_k=2^{k+2} (k<=0), 18 Mc/(k(k+1)) (k>=1);  D=8+18Mc (eq. (7)-(8))."""

    def __init__(self, Mc: float):
        self.Mc = float(Mc)
        self.D = 8.0 + 18.0 * self.Mc

    def sample_k(self, rng):
        if rng.random() < 8.0 / self.D:
            return -int(rng.geometric(0.5) - 1)             # N>=0, P(N=n)=2^{-n-1}
        u = 1.0 - rng.random()
        return int(math.floor(1.0 / u))

    def log_C(self, k):
        logV = (k + 2) * LN2
        if k <= 0:
            return logV - math.log(2 * 2.0 ** (k + 2))
        return logV - math.log(2 * 18.0 * self.Mc / (k * (k + 1.0)))


class PowerEnvelope:
    """Prop 4.9: d_k=V_k (k<=0), 4 M 2^{-eps k} (k>=1) given E Psi^{1+eps} <= M."""

    def __init__(self, M: float, eps: float):
        self.M, self.eps = float(M), float(eps)
        self.Dneg = 8.0
        self.Dpos = 4 * M * 2.0 ** (-eps) / (1 - 2.0 ** (-eps))
        self.D = self.Dneg + self.Dpos

    def sample_k(self, rng):
        if rng.random() < self.Dneg / self.D:
            return -int(rng.geometric(0.5) - 1)
        return int(rng.geometric(1 - 2.0 ** (-self.eps)))     # P(K=k) ∝ 2^{-eps k}, k>=1

    def log_C(self, k):
        logV = (k + 2) * LN2
        if k <= 0:
            return logV - math.log(2 * 2.0 ** (k + 2))
        return logV - math.log(2 * 4 * self.M * 2.0 ** (-self.eps * k))


# --------------------------------------------------------------------------- diagnostics
@dataclass
class Diagnostics:
    race_trials: int = 0
    membership_coins: int = 0
    factory_coins: int = 0
    max_log_C: float = -math.inf
    final_attempts: int = 0
    selected_band: int | None = None
    max_refine_level: int = 0
    completed: bool = False
    trial_bands: list[int] = field(default_factory=list)


@dataclass
class Selection:
    """Finite record of the selected latent object (still refinable)."""

    proposal: ProposalRecord
    band: int
    theta: float
    diagnostics: Diagnostics


def _membership(record: ProposalRecord, k: int, theta: float, diag: Diagnostics, budget: Budget) -> int:
    """Algorithm 3.3: decide (k+theta)ln2 < J < (k+1+theta)ln2 by strict comparisons, refining the record."""
    lo_b, hi_b = (k + theta) * LN2, (k + 1 + theta) * LN2
    n = 1
    while True:
        budget.tick("membership_refine")
        lo, hi = record.interval(n)
        diag.max_refine_level = max(diag.max_refine_level, n)
        if lo > lo_b and hi < hi_b:
            return 1
        if hi < lo_b or lo > hi_b:
            return 0
        n += 1


def _final_accept(record: ProposalRecord, logV: float, rng: np.random.Generator, budget: Budget,
                  diag: Diagnostics) -> bool:
    """Decide U < Psi/V_K, i.e. log U < J - log V_K, with a lazy uniform and refined J intervals."""
    U = LazyUniform(rng)
    n = 1
    while True:
        budget.tick("final_refine")
        lo, hi = record.interval(n)
        diag.max_refine_level = max(diag.max_refine_level, n)
        ulo, uhi = U.value_interval_float()
        # log U in [log ulo, log uhi]
        lulo = math.log(ulo) if ulo > 0 else -math.inf
        luhi = math.log(uhi)
        if luhi < lo - logV:
            return True
        if lulo > hi - logV:
            return False
        U.refine(2)
        n += 1


def retrospective_select(oracle: LogWeightOracle, env: Envelope, rng: np.random.Generator,
                         budget: Budget = NULL_BUDGET, eta: float = 0.5,
                         max_factory_coins: int | None = None) -> Selection:
    """Algorithm 3.6.  Returns the retained accepted proposal (refinable) and diagnostics."""
    diag = Diagnostics()
    theta = float(rng.random())
    while True:
        # ---- countable race, one trial
        while True:
            diag.race_trials += 1
            budget.tick("race_trial")
            k = env.sample_k(rng)
            diag.trial_bands.append(k)
            logC = env.log_C(k)
            diag.max_log_C = max(diag.max_log_C, logC)
            if logC > 0 and (logC > 40 or (max_factory_coins is not None and linear_n0(math.exp(logC), eta) > max_factory_coins)):
                raise BudgetExhausted(f"factory scale C_K=e^{logC:.1f} exceeds work limit (band {k})")
            C = math.exp(logC)
            cache: dict = {}
            crng = rng

            def coin() -> int:
                diag.membership_coins += 1
                rec = oracle.new_proposal(crng)
                bit = _membership(rec, k, theta, diag, budget)
                if bit and "first" not in cache:
                    cache["first"] = rec              # separate cache; the factory sees only the bit
                return bit

            out, used = bernoulli_factory(C, coin, rng, eta=eta, budget=budget) if C > 1 else _thin(C, coin, rng)
            diag.factory_coins += used
            if out:
                break
        # race success: the first-head proposal (exists since output 1 => a head)
        rec = cache["first"]
        diag.selected_band = k
        logV = (k + 2) * LN2
        diag.final_attempts += 1
        if _final_accept(rec, logV, rng, budget, diag):
            diag.completed = True
            return Selection(rec, k, theta, diag)


def _thin(C, coin, rng):
    b = coin()
    return int(b and rng.random() < C), 1


def bounded_weight_select(oracle: LogWeightOracle, log_bound: float, rng: np.random.Generator,
                          budget: Budget = NULL_BUDGET) -> tuple[ProposalRecord, int]:
    """Simpler route when a valid bound Psi<=exp(log_bound) is available: ordinary retrospective rejection."""
    diag = Diagnostics()
    tries = 0
    while True:
        tries += 1
        budget.tick("bounded_try")
        rec = oracle.new_proposal(rng)
        if _final_accept(rec, log_bound, rng, budget, diag):
            return rec, tries
