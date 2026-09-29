"""Paper 3: exact jump-skeleton simulation by localization and conditional reconstruction.

Bounded baseline (§3)                    -> `bounded_skeleton`
Algorithm A (retrospective decision)      -> `retrospective_decision`
Algorithm B (label certification)         -> `layer_of`, `certify_reconstruction`
Algorithm C (next event / survival)       -> `next_event`
Algorithm D (full skeleton, restarts)     -> `simulate_skeleton`

Conditioning kept exactly as in the paper: within one call of C the shift theta, label I and Poisson candidate
schedule eta are FIXED across reconstruction trials; only the whole skeleton + bridge records are redrawn until the
reconstructed latent path has label I.  Restarts after an accepted event use fresh randomness.

Exact outputs: N_T, event times, marks, pre/post states, terminal state.  Continuous segments are only available
through retained bridge records (certified ε-strong enclosures; never exact coordinates).  Bridges of the *unused*
part of the accepted reconstruction are kept as inactive conditioning metadata.

Every primitive call is counted in a `Budget`; exhausting an explicit limit raises `BudgetExhausted` (the run must
then be recorded as censored, not dropped).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..rng import NULL_BUDGET, Budget, BudgetExhausted, LazyUniform


# --------------------------------------------------------------------------------------------- data containers
@dataclass
class Event:
    tau: float
    pre: np.ndarray
    mark: Any
    post: np.ndarray


@dataclass
class SegmentInfo:
    theta: float = float("nan")
    label: int = -1
    ell: float = float("nan")
    n_candidates: int = 0
    skeleton_trials: int = 0
    skeleton_early_rejects: int = 0
    bridges_built: int = 0
    label_refinements: int = 0
    forward_label_levels: int = 0
    accepted_candidate: int = 0
    wall: float = 0.0
    route: str = "localized"


@dataclass
class Segment:
    """One physical inter-event segment [s, t_end] with the retained latent-path records."""

    s: float
    x: np.ndarray                         # post-event (or initial) exact state at s
    t_end: float
    x_end: np.ndarray                     # exact pre-event or terminal state at t_end
    anchor_times: list[float]             # candidate times (auxiliary; NOT part of the target output law)
    anchor_states: list[np.ndarray]
    bridges: list                         # active bridge records covering [s,t_end]
    inactive: list = field(default_factory=list)   # conditioning metadata of the unused hypothetical continuation
    info: SegmentInfo = field(default_factory=SegmentInfo)


@dataclass
class PathApprox:
    """Right-continuous piecewise approximation of the retained latent process, uniform error <= radius."""

    pieces: list                          # list of (times, values) per segment
    radius: float
    jump_times: list[float]

    def __call__(self, t: float) -> np.ndarray:
        for k, (ts, vs) in enumerate(self.pieces):
            last = k == len(self.pieces) - 1
            if ts[0] <= t < ts[-1] or (last and t <= ts[-1]):
                return np.array([np.interp(t, ts, vs[:, j]) for j in range(vs.shape[1])])
        raise ValueError("t outside [0,T]")

    def grid(self, n: int = 400):
        T = self.pieces[-1][0][-1]
        tt = np.linspace(0, T, n)
        return tt, np.array([self(t) for t in tt])


@dataclass
class JumpSkeleton:
    """Output S_T of eq. (3) + retained records.  `events`, `terminal`, `n_events` are the exact skeleton."""

    x0: np.ndarray
    T: float
    events: list[Event]
    terminal: np.ndarray
    segments: list[Segment]
    meta: dict = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def refine(self, eps: float) -> PathApprox:
        """Finite-description approximation of the latent càdlàg process with sup error <= eps per segment
        (the SAME latent paths are refined; bridge records are extended, never resampled)."""
        pieces, radius = [], 0.0
        for seg in self.segments:
            ts, vs = [], []
            for rec in seg.bridges:
                enc = rec.refine_to(eps)
                radius = max(radius, enc.radius)
                if ts:
                    ts.pop(), vs.pop()
                ts += list(enc.times)
                vs += list(enc.values)
            if not seg.bridges:                       # degenerate zero-length segment
                ts, vs = [seg.s, seg.t_end], [seg.x, seg.x_end]
            pieces.append((np.array(ts), np.array(vs)))
        return PathApprox(pieces, radius, [e.tau for e in self.events])


# --------------------------------------------------------------------------------------------- Algorithm A
def retrospective_decision(L: float, enclosure: Callable[[int], tuple[float, float]], rng: np.random.Generator,
                           budget: Budget = NULL_BUDGET) -> bool:
    """Exact 1{L U < Q} for a latent Q in [0,L] with nested certified intervals `enclosure(n)`; U fresh uniform."""
    assert L > 0
    U = LazyUniform(rng)
    n = 1
    lo_run, hi_run = 0.0, L
    while True:
        budget.tick("decision_refine")
        lo, hi = enclosure(n)
        lo_run, hi_run = max(lo_run, lo), min(hi_run, hi)
        ulo, uhi = U.value_interval_float()
        if L * uhi < lo_run:
            return True
        if L * ulo > hi_run:
            return False
        U.refine(2)
        n += 1


# --------------------------------------------------------------------------------------------- Algorithm B
def radius_of(i: int, theta: float) -> float:
    return 2.0 ** (i + theta)


def layer_of(x: float, theta: float) -> int:
    """L^theta for a known value: min{i>=0: x < 2^{i+theta}}."""
    if x < 2.0**theta:
        return 0
    return int(math.floor(math.log2(x) - theta)) + 1


def certify_reconstruction(records: list, theta: float, target: int, budget: Budget = NULL_BUDGET,
                           info: SegmentInfo | None = None) -> bool:
    """Decide (lazily, exactly) whether the concatenated latent path of `records` has label == target.
    Refines only what is needed; rejection is reported as soon as some segment certifiably exceeds r_target or all
    upper bounds fall below r_{target-1}.  Never resamples a record."""
    n = [1] * len(records)
    a = [0.0] * len(records)
    b = [math.inf] * len(records)
    for j, rec in enumerate(records):
        lo, hi = rec.max_norm_interval(n[j])
        a[j], b[j] = lo, hi
    while True:
        A, B = max(a), max(b)
        iA, iB = layer_of(A, theta), layer_of(B, theta)
        if iA == iB:
            return iA == target
        if iA > target or iB < target:
            return False
        # ambiguous: refine the record(s) that decide the maximum
        cand = [j for j in range(len(records)) if b[j] >= A]
        j = max(cand, key=lambda k: b[k] - a[k])
        n[j] += 1
        budget.tick("label_refine")
        if info is not None:
            info.label_refinements += 1
        lo, hi = records[j].max_norm_interval(n[j])
        a[j], b[j] = max(a[j], lo), min(b[j], hi)


def forward_label(record, theta: float, budget: Budget = NULL_BUDGET) -> tuple[int, int]:
    """Algorithm B on one retained forward record.  Returns (label, refinement level used)."""
    n = 1
    a, b = 0.0, math.inf
    while True:
        budget.tick("label_refine")
        lo, hi = record.max_norm_interval(n)
        a, b = max(a, lo), min(b, hi)
        iA, iB = layer_of(a, theta), layer_of(b, theta)
        if iA == iB:
            return iA, n
        n += 1


# --------------------------------------------------------------------------------------------- Algorithm C
@dataclass
class EventResult:
    kind: str                 # 'event' | 'survival'
    tau: float | None
    pre: np.ndarray | None
    mark: Any
    post: np.ndarray | None
    terminal: np.ndarray | None
    segment: Segment


def _poisson_schedule(rate: float, s: float, T: float, rng: np.random.Generator) -> np.ndarray:
    if rate <= 0:
        return np.empty(0)
    M = int(rng.poisson(rate * (T - s)))
    return np.sort(s + (T - s) * rng.random(M))


def localize(diffusion, bridge_factory, s: float, x: np.ndarray, T: float, rng: np.random.Generator,
             budget: Budget = NULL_BUDGET) -> tuple[float, int, int]:
    """Steps 1 of Algorithm C: theta, and the label I of an independent forward record.  Returns (theta, I, level)."""
    theta = float(rng.random())
    yT = diffusion.sample_transition(s, x, T, rng)
    budget.tick("transition")
    fwd = bridge_factory.new_bridge(s, x, T, yT, rng)
    I, lev = forward_label(fwd, theta, budget)
    return theta, I, lev            # forward record released here; only (theta, I) are carried over


def reconstruct(diffusion, bridge_factory, s: float, x: np.ndarray, T: float, times: np.ndarray, theta: float,
                I: int, rng: np.random.Generator, budget: Budget = NULL_BUDGET, use_filter: bool = True,
                info: SegmentInfo | None = None):
    """Conditional reconstruction (Lemma 4.2): with (theta, I, schedule=times) FIXED, redraw the whole exact skeleton and
    independent bridge records until the latent concatenated path has label I.  Returns (states, records)."""
    info = info if info is not None else SegmentInfo()
    rI = radius_of(I, theta)
    M = len(times) - 2
    while True:
        info.skeleton_trials += 1
        budget.tick("skeleton_trial")
        z = [x]
        rejected = False
        if use_filter and np.linalg.norm(x) >= rI:
            rejected = True
        else:
            for j in range(1, M + 2):
                zj = diffusion.sample_transition(times[j - 1], z[-1], times[j], rng)
                budget.tick("transition")
                z.append(zj)
                if use_filter and np.linalg.norm(zj) >= rI:            # optional whole-skeleton filter (lazy)
                    rejected = True
                    break
        if rejected:
            info.skeleton_early_rejects += 1
            continue
        recs = []
        for j in range(M + 1):
            recs.append(bridge_factory.new_bridge(times[j], z[j], times[j + 1], z[j + 1], rng))
            info.bridges_built += 1
        if certify_reconstruction(recs, theta, I, budget, info):
            return z, recs


def next_event(diffusion, jump, bridge_factory, s: float, x: np.ndarray, T: float, rng: np.random.Generator,
               budget: Budget = NULL_BUDGET, use_filter: bool = True, keep_records: bool = True) -> EventResult:
    """Algorithm C: first event after s (with exact pre/post states) or survival to T with exact X_T."""
    t_start = time.perf_counter()
    info = SegmentInfo()
    x = np.asarray(x, float)
    theta, I, lev = localize(diffusion, bridge_factory, s, x, T, rng, budget)
    info.theta, info.label, info.forward_label_levels = theta, I, lev
    ell = float(jump.local_bound(s, T, radius_of(I, theta)))
    info.ell = ell
    eta = _poisson_schedule(ell, s, T, rng)           # fixed across reconstruction trials
    info.n_candidates = M = len(eta)
    times = np.concatenate([[s], eta, [T]])
    z, recs = reconstruct(diffusion, bridge_factory, s, x, T, times, theta, I, rng, budget, use_filter, info)
    for j in range(1, M + 1):                          # thinning scan with exact candidate states
        lam = float(jump.intensity(times[j], z[j]))
        budget.tick("candidate")
        assert lam <= ell * (1 + 1e-9) + 1e-12, "local bound violated"
        if LazyUniform(rng).compare_value(lam / ell):
            mark = jump.sample_mark(times[j], z[j], rng)
            post = np.asarray(jump.update(times[j], z[j], mark), float)
            info.accepted_candidate = j
            info.wall = time.perf_counter() - t_start
            seg = Segment(s, x, float(times[j]), z[j], [float(t) for t in times[: j + 1]], z[: j + 1],
                          recs[:j] if keep_records else [], recs[j:] if keep_records else [], info)
            return EventResult("event", float(times[j]), z[j], mark, post, None, seg)
    info.wall = time.perf_counter() - t_start
    seg = Segment(s, x, T, z[-1], [float(t) for t in times], z, recs if keep_records else [], [], info)
    return EventResult("survival", None, None, None, None, z[-1], seg)


# --------------------------------------------------------------------------------------------- Algorithm D
def simulate_skeleton(diffusion, jump, bridge_factory, x0, T: float, rng: np.random.Generator,
                      budget: Budget = NULL_BUDGET, use_filter: bool = True, max_events: int = 10_000,
                      keep_records: bool = True, checkpoint: Callable[[dict], None] | None = None,
                      resume: dict | None = None) -> JumpSkeleton:
    """Algorithm D.  `checkpoint(state)` is called before every next-event call with the exact resumable state
    (s, x, events, rng bit-generator state); `resume` restarts from such a state and reproduces the same realisation."""
    x0 = np.asarray(x0, float)
    if T == 0:
        return JumpSkeleton(x0, 0.0, [], x0, [], {"n_calls": 0})
    if resume is None:
        s, x, events, segments = 0.0, x0.copy(), [], []
    else:
        s, x, events, segments = resume["s"], np.asarray(resume["x"], float), list(resume["events"]), list(resume["segments"])
        rng.bit_generator.state = resume["rng_state"]
    calls = 0
    while s < T:
        if checkpoint is not None:
            checkpoint(dict(s=s, x=x.copy(), events=list(events), segments=list(segments),
                            rng_state=rng.bit_generator.state))
        res = next_event(diffusion, jump, bridge_factory, s, x, T, rng, budget, use_filter, keep_records)
        calls += 1
        segments.append(res.segment)
        if res.kind == "survival":
            return JumpSkeleton(x0, T, events, res.terminal, segments, {"n_calls": calls})
        events.append(Event(res.tau, res.pre, res.mark, res.post))
        s, x = res.tau, res.post
        if len(events) > max_events:
            raise BudgetExhausted(f"more than {max_events} events (nonexplosion assumption violated or huge rate)")
    return JumpSkeleton(x0, T, events, x, segments, {"n_calls": calls})


# --------------------------------------------------------------------------------------------- bounded baseline
def bounded_skeleton(diffusion, jump, x0, T: float, rng: np.random.Generator, budget: Budget = NULL_BUDGET,
                     max_events: int = 10_000) -> JumpSkeleton:
    """§3 bounded-intensity thinning: candidate times from a rate-L Poisson process (L=jump.global_bound), exact
    transitions between candidates, accept with probability lambda/L.  No bridge records (attach later if desired)."""
    L = float(jump.global_bound)
    x = np.asarray(x0, float)
    events: list[Event] = []
    if T == 0:
        return JumpSkeleton(x, 0.0, [], x, [], {})
    cand = _poisson_schedule(L, 0.0, T, rng)
    s = 0.0
    for tj in cand:
        y = diffusion.sample_transition(s, x, float(tj), rng)
        budget.tick("transition")
        s = float(tj)
        lam = float(jump.intensity(s, y))
        budget.tick("candidate")
        assert lam <= L * (1 + 1e-12)
        if LazyUniform(rng).compare_value(lam / L):
            z = jump.sample_mark(s, y, rng)
            x_new = np.asarray(jump.update(s, y, z), float)
            events.append(Event(s, y, z, x_new))
            x = x_new
        else:
            x = y
        if len(events) > max_events:
            raise BudgetExhausted("too many events")
    x = diffusion.sample_transition(s, x, T, rng)
    budget.tick("transition")
    return JumpSkeleton(np.asarray(x0, float), T, events, x, [], {"n_candidates": len(cand)})
