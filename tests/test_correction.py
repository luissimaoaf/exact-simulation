import math

import numpy as np
import pytest
from scipy.special import polygamma
from scipy.stats import norm

from exactjd.correction import factory as fac
from exactjd.correction.abstract import (LogEnvelope, PowerEnvelope, bounded_weight_select, retrospective_select)
from exactjd.correction import moment as mom
from exactjd.models.ou import OU
from exactjd.rng import Budget, BudgetExhausted


# ------------------------------------------------------------------ factory
@pytest.mark.parametrize("C", [1.5, 4.0, 30.0])
def test_factory_increment_bound_d_le_pi(C):
    eta = 0.5
    n0 = fac.linear_n0(C, eta)
    tot = float(polygamma(1, n0))
    worst = 0.0
    for m in list(range(n0 + 1, n0 + 400)) + [n0 + 5000, n0 + 100000]:
        s = np.arange(1, m + 1)
        lam = s / m
        d = fac._fhat(s / m, C, eta) - (lam * fac._fhat((s - 1) / (m - 1), C, eta) + (1 - lam) * fac._fhat(s / (m - 1), C, eta))
        pi = (eta / 2) * (m - 1.0) ** -2 / tot
        assert np.all(d >= -1e-12)
        worst = max(worst, float(np.max(d) / pi))
    assert worst <= 1.0 + 1e-9


@pytest.mark.parametrize("C,p", [(3.0, 0.1), (3.0, 0.16), (1.7, 0.29)])
def test_linear_factory_law_and_head_property(C, p):
    rng = np.random.default_rng(int(C * 100 + p * 1000))
    n, heads_needed_violation = 12000, 0
    outs, coins = [], []
    for _ in range(n):
        seen = {"h": 0}

        def coin():
            b = int(rng.random() < p)
            seen["h"] += b
            return b
        o, used = fac.linear_factory(C, 0.5, coin, rng)
        if o == 1 and seen["h"] == 0:
            heads_needed_violation += 1
        outs.append(o)
        coins.append(used)
    outs = np.array(outs)
    assert abs(outs.mean() - C * p) < 4 * math.sqrt(C * p * (1 - C * p) / n)
    assert heads_needed_violation == 0
    assert np.mean(coins) >= C           # Lemma 4.1 lower bound


def test_linear_factory_zero_bias_outputs_zero():
    rng = np.random.default_rng(0)
    for _ in range(300):
        o, _ = fac.linear_factory(5.0, 0.5, lambda: 0, rng)
        assert o == 0


def test_thinning_factory():
    rng = np.random.default_rng(1)
    outs = [fac.thinning_factory(0.4, lambda: int(rng.random() < 0.3), rng)[0] for _ in range(30000)]
    assert abs(np.mean(outs) - 0.12) < 0.008


# ------------------------------------------------------------------ envelopes
def test_log_envelope_sampler_matches_d_over_D():
    Mc = 1.0
    env = LogEnvelope(Mc)
    rng = np.random.default_rng(2)
    ks = np.array([env.sample_k(rng) for _ in range(60000)])
    D = 8 + 18 * Mc
    for k in (1, 2, 5):
        assert (ks == k).mean() == pytest.approx(18 * Mc / (k * (k + 1)) / D, abs=0.004)
    assert (ks <= 0).mean() == pytest.approx(8 / D, abs=0.005)
    assert (ks >= 24).mean() == pytest.approx(18 * Mc / (24 * D), abs=0.003)     # Prop 4.8 tail
    assert env.log_C(24) == pytest.approx(math.log(1.1185e9), rel=1e-3)          # paper's numerical diagnostic for Mc=1


def test_power_envelope_valid_scale():
    env = PowerEnvelope(M=2.0, eps=1.0)
    rng = np.random.default_rng(3)
    ks = np.array([env.sample_k(rng) for _ in range(20000)])
    assert (ks >= 1).mean() == pytest.approx(env.Dpos / env.D, abs=0.01)


# ------------------------------------------------------------------ abstract race on the lognormal example
class GaussTiltProposal:
    def __init__(self, c, T, rng):
        R = math.sqrt(T) * rng.standard_normal()
        self.R, self.J = R, -c * R - c * c * T

    def interval(self, n):
        w = 3.0 * 2.0 ** (-n)
        return self.J - w, self.J + w


class GaussTiltOracle:
    def __init__(self, c=1.0, T=1.0):
        self.c, self.T = c, T

    def new_proposal(self, rng):
        return GaussTiltProposal(self.c, self.T, rng)


class TruncatedLogEnvelope(LogEnvelope):
    """TEST HARNESS ONLY: support truncated to bands kmin..kmax (bias ~ P(band outside) < 1e-6 here).
    Never used for exactness claims; the paper's fixed envelope has an infinite-mean race (Thm 4.2)."""

    def __init__(self, Mc, kmin=-12, kmax=6):
        super().__init__(Mc)
        self.kmin, self.kmax = kmin, kmax

    def sample_k(self, rng):
        while True:
            k = super().sample_k(rng)
            if self.kmin <= k <= self.kmax:
                return k


@pytest.mark.slow
def test_race_selects_tilted_gaussian_law():
    c, T = 1.0, 1.0
    Mc = math.exp(-c * c * T / 2) * 2.4
    env = TruncatedLogEnvelope(Mc)
    orc = GaussTiltOracle(c, T)
    rng = np.random.default_rng(11)
    n = 1500
    R, diags = [], []
    for _ in range(n):
        sel = retrospective_select(orc, env, rng)
        R.append(sel.proposal.R)
        diags.append(sel.diagnostics)
    R = np.array(R)
    assert abs(R.mean() - (-c * T)) < 4 * math.sqrt(T / n)          # target N(-cT, T)
    assert R.var() == pytest.approx(T, rel=0.12)
    trials = np.mean([d.race_trials for d in diags])
    Z = math.exp(-c * c * T / 2)
    # Prop 4.4 : mean raw trials 2 D_trunc / Z (D_trunc <= D); order-of-magnitude check
    assert 0.2 * (16 + 36 * Mc) / Z < trials < 1.3 * (16 + 36 * Mc) / Z


def test_bounded_weight_route_law():
    rng = np.random.default_rng(5)

    class P:
        def __init__(self, r):
            x = r.standard_normal()
            self.x, self.J = x, math.sin(x)

        def interval(self, n):
            w = 2.0 ** (-n)
            return self.J - w, self.J + w

    class O:
        def new_proposal(self, r):
            return P(r)

    xs = np.array([bounded_weight_select(O(), 1.0, rng)[0].x for _ in range(8000)])
    g = np.linspace(-9, 9, 200001)
    w = np.exp(np.sin(g)) * norm.pdf(g)
    mean = np.trapezoid(g * w, g) / np.trapezoid(w, g)
    assert abs(xs.mean() - mean) < 4 * xs.std() / math.sqrt(len(xs))


def test_paper_envelope_reports_censoring_not_silent_truncation():
    """With the fixed log envelope, huge factory scales appear; a work limit must RAISE, never truncate silently."""
    orc = GaussTiltOracle(1.0, 1.0)
    env = LogEnvelope(1.0)
    rng = np.random.default_rng(0)
    raised = 0
    for _ in range(40):
        try:
            retrospective_select(orc, env, rng, max_factory_coins=2000)
        except BudgetExhausted:
            raised += 1
    assert raised > 0


# ------------------------------------------------------------------ moment certificates
def test_tight_certificate_matches_monte_carlo_moments():
    ou = OU(np.array([[-1.0, 0.3], [-0.2, -0.5]]), [0.2, 0.0], np.eye(2))
    T, u, v = 0.5, np.zeros(2), np.array([0.4, -0.3])
    panels = 10
    t, w = mom._graded_nodes(T, panels)
    res = mom.ou_bridge_J_second_moment(ou, T, u, v, panels=panels)
    S = np.linalg.inv(ou.a)
    rng = np.random.default_rng(4)
    J = []
    for _ in range(2500):
        X = ou.sample_bridge_values(0.0, u, T, v, t, rng)
        G = -np.einsum("ij,jk,ik->i", X @ ou.B.T + ou.c, S, X - v) / (T - t)
        J.append(float(w @ G))
    J = np.array(J)
    assert abs(J.mean() - res["EJ"]) < 4 * J.std() / math.sqrt(len(J)) + 1e-3
    assert (J**2).mean() == pytest.approx(res["EJ2"], rel=0.12)


def test_certificates_ordering():
    ou = OU(np.array([[-1.0, 0.3], [-0.2, -0.5]]), [0.2, 0.0], np.eye(2))
    T, u, v = 0.25, np.zeros(2), np.array([0.3, -0.2])
    tight = mom.ou_tight_certificate(ou, T, u, v)
    hk = mom.ou_heat_kernel_constants(ou, T)
    th = mom.ou_endpoint_data(ou, T, u, v)["theta_uT"]
    thm84 = mom.layer_certificate(2, T, u, v, ou.a, hk, th)
    assert tight["Z"] <= thm84["Z_plus"]
    assert tight["M_c"] < thm84["M_c"]
