import math

import numpy as np
import pytest
from scipy import stats

from exactjd.models.ou import OU
from exactjd.models.jump import ConstantRateJump, BoundedStateRateJump, QuadraticRateJump
from exactjd.jump.records import OUBridgeFactory
from exactjd.jump import skeleton as sk
from exactjd.rng import Budget, BudgetExhausted

OU2 = OU.isotropic(1.0, np.eye(2))
FAC = OUBridgeFactory(OU2)


class ZeroRate:
    d = 2
    global_bound = 0.0

    def intensity(self, t, x):
        return 0.0

    def local_bound(self, s, T, r):
        return 0.0

    def sample_mark(self, t, x, rng):
        return np.zeros(2)

    def update(self, t, x, z):
        return x


# ------------------------------------------------------------------ Algorithm A / B primitives
def test_layer_of_boundaries():
    th = 0.3
    assert sk.layer_of(2**th * 0.99, th) == 0
    assert sk.layer_of(2**th * 1.01, th) == 1
    assert sk.layer_of(2 ** (3 + th) * 0.999, th) == 3
    assert sk.layer_of(2 ** (3 + th) * 1.001, th) == 4


def test_retrospective_decision_unbiased():
    rng = np.random.default_rng(0)
    Q, L = 0.37, 1.3
    n = 12000
    hits = sum(sk.retrospective_decision(L, lambda k: (Q - 2.0**-k, Q + 2.0**-k), rng) for _ in range(n))
    assert abs(hits / n - Q / L) < 4 * math.sqrt(Q / L * (1 - Q / L) / n)


def test_sup_intervals_contain_fine_reference_and_nested():
    rng = np.random.default_rng(1)
    for i in range(25):
        x = rng.normal(size=2) * 0.7
        y = OU2.sample_transition(0.0, x, 0.8, rng)
        rec = FAC.new_bridge(0.0, x, 0.8, y, rng)
        enc = rec.enclosure(15)
        ref = float(np.max(np.linalg.norm(enc.values, axis=1)))      # proxy for the latent sup (level-15 vertices)
        prev = (0.0, math.inf)
        for n in range(1, 12):
            lo, hi = rec.max_norm_interval(n)
            assert lo >= prev[0] - 1e-12 and hi <= prev[1] + 1e-12
            assert lo <= ref + rec.enclosure(15).radius and hi >= ref - 1e-9
            prev = (lo, hi)
        assert prev[1] - prev[0] < 1.5


def test_ou_bridge_record_endpoint_and_lazy_consistency():
    """Lazily drawn cell noises must agree with later global levels (same latent path)."""
    rng = np.random.default_rng(2)
    x, y = np.array([0.2, -0.1]), np.array([0.5, 0.4])
    rec = FAC.new_bridge(0.0, x, 1.0, y, rng)
    rec.max_norm_interval(6)                      # adaptive drawing first
    nodes_before = {k: v.copy() for k, v in rec.rec._axi.items()}
    assert len(nodes_before) > 0
    e = rec.enclosure(10)                         # then global arrays: must honour the lazy entries
    assert np.allclose(e.values[0], x) and np.allclose(e.values[-1], y)
    checked = 0
    for (k, idx), v in nodes_before.items():
        if k <= 10:
            assert np.array_equal(rec.rec._xi[k - 1][idx], v)
            checked += 1
    assert checked > 0


# ------------------------------------------------------------------ Lemma 4.2 : reconstruction law
@pytest.mark.slow
def test_reconstruction_preserves_unconditional_law():
    """Fixed schedule; (theta, I) from an independent forward record; accepted reconstruction must have the diffusion's
    law at the candidate times and at T (Lemma 4.2 / eq. (12) marginal)."""
    rng = np.random.default_rng(3)
    x0, s, T = np.array([0.4, -0.3]), 0.0, 1.0
    times = np.array([0.0, 0.3, 0.6, 1.0])
    n = 1500
    z1, z3 = [], []
    for _ in range(n):
        theta, I, _ = sk.localize(OU2, FAC, s, x0, T, rng)
        z, recs = sk.reconstruct(OU2, FAC, s, x0, T, times, theta, I, rng)
        z1.append(z[1])
        z3.append(z[3])
    z1, z3 = np.array(z1), np.array(z3)
    for zz, t in ((z1, 0.3), (z3, 1.0)):
        mu, Q = OU2.mean_cov(t, x0)
        se = np.sqrt(np.diag(Q) / n)
        assert np.all(np.abs(zz.mean(0) - mu) < 4.2 * se)
        assert np.allclose(np.cov(zz.T), Q, atol=0.07)


# ------------------------------------------------------------------ bounded baseline and constant-rate analytics
def const_ref(T, x0, lam, jumpsize, kappa=1.0):
    mean = math.exp(-kappa * T) * np.asarray(x0)
    q = (1 - math.exp(-2 * kappa * T)) / (2 * kappa)
    var = q + lam * jumpsize**2 * q                 # int_0^T e^{-2k(T-s)} ds = q
    return mean, var


def test_bounded_baseline_constant_rate_analytics():
    jump = ConstantRateJump(2, lam=2.0, jump=0.2)
    rng = np.random.default_rng(4)
    n, T = 6000, 1.0
    x0 = np.array([0.5, -0.5])
    res = [sk.bounded_skeleton(OU2, jump, x0, T, rng) for _ in range(n)]
    N = np.array([r.n_events for r in res])
    xt = np.array([r.terminal for r in res])
    assert abs(N.mean() - 2.0) < 4 * math.sqrt(2.0 / n)
    pk = np.append(stats.poisson.pmf(np.arange(0, 7), 2.0), stats.poisson.sf(6, 2.0))
    obs = np.array([(N == k).sum() for k in range(7)] + [(N > 6).sum()])
    chi = stats.chisquare(obs, pk * n)
    assert chi.pvalue > 1e-3
    mean, var = const_ref(T, x0, 2.0, 0.2)
    assert np.allclose(xt.mean(0), mean, atol=4 * math.sqrt(var / n))
    assert np.allclose(xt.var(0), var, rtol=0.07)


@pytest.mark.slow
def test_localized_constant_rate_matches_analytics():
    jump = ConstantRateJump(2, lam=2.0, jump=0.2)
    rng = np.random.default_rng(5)
    n, T = 1500, 1.0
    x0 = np.array([0.5, -0.5])
    res = [sk.simulate_skeleton(OU2, jump, FAC, x0, T, rng, keep_records=False) for _ in range(n)]
    N = np.array([r.n_events for r in res])
    xt = np.array([r.terminal for r in res])
    assert abs(N.mean() - 2.0) < 4 * math.sqrt(2.0 / n)
    mean, var = const_ref(T, x0, 2.0, 0.2)
    assert np.allclose(xt.mean(0), mean, atol=4 * math.sqrt(var / n))
    assert np.allclose(xt.var(0), var, rtol=0.12)


# ------------------------------------------------------------------ unbounded intensity: analytic first-event survival
@pytest.mark.slow
def test_first_event_survival_quadratic_intensity():
    lam0, lam1 = 1.0, 1.0
    jump = QuadraticRateJump(2, lam0, lam1, rho=0.5)
    rng = np.random.default_rng(6)
    x0, T, n = np.array([0.5, 0.5]), 1.0, 1400
    tau1 = []
    for _ in range(n):
        r = sk.next_event(OU2, jump, FAC, 0.0, x0, T, rng, keep_records=False)
        tau1.append(r.tau if r.kind == "event" else math.inf)
    tau1 = np.array(tau1)
    for t in (0.25, 0.5, 1.0):
        emp = (tau1 > t).mean()
        ref = OU2.survival_quadratic_rate(t, x0, lam0, lam1)
        assert abs(emp - ref) < 4 * math.sqrt(ref * (1 - ref) / n) + 1e-3, (t, emp, ref)


@pytest.mark.slow
def test_bounded_state_dependent_localized_matches_global_thinning():
    jump = BoundedStateRateJump(2, rho=0.5)
    rng = np.random.default_rng(7)
    x0, T, n = np.array([0.3, 0.3]), 1.0, 1000
    a = np.array([sk.simulate_skeleton(OU2, jump, FAC, x0, T, rng, keep_records=False).n_events for _ in range(n)])
    b = np.array([sk.bounded_skeleton(OU2, jump, x0, T, rng).n_events for _ in range(n)])
    ks = np.arange(0, 9)
    oa = np.array([(a == k).sum() for k in ks] + [(a > 8).sum()])
    ob = np.array([(b == k).sum() for k in ks] + [(b > 8).sum()])
    keep = (oa + ob) >= 10
    tab = np.vstack([oa[keep], ob[keep]])
    assert stats.chi2_contingency(tab)[1] > 1e-3


# ------------------------------------------------------------------ edge cases
def test_zero_intensity_no_events_and_unconditional_terminal_law():
    rng = np.random.default_rng(8)
    x0, T, n = np.array([0.4, 0.1]), 0.7, 1500
    sks = [sk.simulate_skeleton(OU2, ZeroRate(), FAC, x0, T, rng, keep_records=False) for _ in range(n)]
    assert all(s.n_events == 0 for s in sks)
    xt = np.array([s.terminal for s in sks])
    mu, Q = OU2.mean_cov(T, x0)
    assert np.all(np.abs(xt.mean(0) - mu) < 4.2 * np.sqrt(np.diag(Q) / n))
    assert np.allclose(np.cov(xt.T), Q, atol=0.05)


def test_T_zero_and_tiny_horizon_and_random_start():
    rng = np.random.default_rng(9)
    jump = QuadraticRateJump(2, 1.0, 1.0, 0.5)
    s0 = sk.simulate_skeleton(OU2, jump, FAC, np.array([1.0, 2.0]), 0.0, rng)
    assert s0.n_events == 0 and np.allclose(s0.terminal, [1.0, 2.0])
    for T in (1e-6, 1e-3, 0.05):
        s1 = sk.simulate_skeleton(OU2, jump, FAC, rng.normal(size=2), T, rng)
        assert 0 <= s1.n_events <= 5
        assert all(0 < e.tau < T for e in s1.events)


def test_dimension_one_and_arbitrary_positive_segments():
    ou1 = OU.isotropic(0.7, np.array([[1.3]]))
    fac1 = OUBridgeFactory(ou1)
    jump = QuadraticRateJump(1, 0.5, 2.0, 0.3)
    rng = np.random.default_rng(10)
    for T in (0.37, 2.5):
        s = sk.simulate_skeleton(ou1, jump, fac1, np.array([0.2]), T, rng)
        assert s.terminal.shape == (1,)
        times = [e.tau for e in s.events]
        assert times == sorted(times)


# ------------------------------------------------------------------ refinement, resume, budgets
def test_repeated_refinement_of_completed_realisation():
    rng = np.random.default_rng(11)
    jump = QuadraticRateJump(2, 2.0, 1.0, 0.5)
    s = sk.simulate_skeleton(OU2, jump, FAC, np.array([0.3, -0.2]), 1.0, rng)
    p1 = s.refine(0.3)
    p2 = s.refine(0.1)
    assert p1.radius <= 0.3 and p2.radius <= 0.1 and p2.radius <= p1.radius
    tt, v1 = p1.grid(300)
    _, v2 = p2.grid(300)
    assert np.max(np.linalg.norm(v1 - v2, axis=1)) <= p1.radius + p2.radius + 1e-9
    for e in s.events:                                         # exact pre/post states are respected
        assert np.linalg.norm(p2(e.tau) - e.post) < 1e-9
        assert np.linalg.norm(p2(e.tau - 1e-7) - e.pre) < 0.1 + 1e-3
    assert np.linalg.norm(p2(1.0) - s.terminal) < 1e-9
    assert np.allclose(p2(0.0), [0.3, -0.2])


def test_checkpoint_resume_reproduces_realisation():
    jump = QuadraticRateJump(2, 3.0, 1.0, 0.5)
    states = []
    rng = np.random.default_rng(12)
    full = sk.simulate_skeleton(OU2, jump, FAC, np.array([0.5, 0.5]), 1.0, rng,
                                checkpoint=lambda st: states.append(st))
    assert len(states) >= 2 or full.n_events == 0
    k = min(1, len(states) - 1)
    rng2 = np.random.default_rng(999)                       # unrelated generator: state is restored from checkpoint
    res = sk.simulate_skeleton(OU2, jump, FAC, np.array([0.5, 0.5]), 1.0, rng2, resume=states[k])
    assert res.n_events == full.n_events
    for a, b in zip(res.events, full.events):
        assert a.tau == b.tau and np.array_equal(a.post, b.post) and np.array_equal(a.mark, b.mark)
    assert np.array_equal(res.terminal, full.terminal)


def test_budget_exhaustion_is_explicit():
    jump = QuadraticRateJump(2, 5.0, 2.0, 0.5)
    with pytest.raises(BudgetExhausted):
        sk.simulate_skeleton(OU2, jump, FAC, np.array([1.0, 1.0]), 1.0, np.random.default_rng(0), budget=Budget(max_calls=8))


@pytest.mark.slow
def test_restarts_use_fresh_independent_randomness():
    """After an accepted event the next call draws a fresh shift/label/schedule (no reuse of the old continuation)."""
    jump = QuadraticRateJump(2, 4.0, 2.0, 0.5)
    rng = np.random.default_rng(21)
    pairs, labels = [], []
    for _ in range(90):
        s = sk.simulate_skeleton(OU2, jump, FAC, np.array([0.5, 0.5]), 1.0, rng, keep_records=False)
        infos = [g.info for g in s.segments]
        pairs += [(a.theta, b.theta) for a, b in zip(infos[:-1], infos[1:])]
        labels += [g.label for g in infos]
    a, b = np.array(pairs).T
    assert len(a) > 100
    assert abs(np.corrcoef(a, b)[0, 1]) < 4 / math.sqrt(len(a))
    assert stats.kstest(np.concatenate([a, b]), "uniform").pvalue > 1e-3


def test_state_dependent_marks_use_the_exact_pre_event_state():
    from exactjd.models.jump import StateDependentMarkJump
    jump = StateDependentMarkJump(2, 3.0, 1.0, 0.5, jump=0.3)
    rng = np.random.default_rng(31)
    seen = 0
    for _ in range(25):
        s = sk.simulate_skeleton(OU2, jump, FAC, np.array([0.8, 0.4]), 1.0, rng, keep_records=False)
        for e in s.events:
            assert np.allclose(np.abs(e.mark), 0.3 * math.tanh(np.linalg.norm(e.pre)))
            assert np.allclose(e.post, 0.5 * e.pre + e.mark)
            seen += 1
    assert seen > 10
