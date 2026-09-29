import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from exactjd.models.ou import OU
from exactjd.bridges.additive import AdditiveModel, AdditiveGuidedBridge
from exactjd.correction.bridge import GuidedProposal, OUGuidedOracle
from exactjd.correction import moment as mom

OU2 = OU(np.array([[-1.0, 0.3], [-0.2, -0.5]]), [0.2, 0.0], np.eye(2))
U0, V0, T0 = np.zeros(2), np.array([1.0, -1.0]), 1.0


def make(seed, **kw):
    return AdditiveGuidedBridge(AdditiveModel.from_ou(OU2), U0, V0, T0, np.random.default_rng(seed), **kw)


def test_rk4_and_affine_centers_agree():
    a = make(3, fast=True)
    b = make(3, fast=False)
    for n in (5, 8):
        ea, eb = a.enclosure(n), b.enclosure(n)
        k = min(len(ea.times), len(eb.times))
        assert np.allclose(ea.values[: 2 ** n], eb.values[: 2 ** n], atol=1e-6)


def test_pinned_and_nested_radius_decreases():
    br = make(1)
    radii = [br.enclosure(n).radius for n in (6, 8, 10, 12)]
    assert all(radii[i] > radii[i + 1] for i in range(3))
    e = br.enclosure(10)
    assert np.allclose(e(0.0), U0) and np.allclose(e(T0), V0)
    parts = e.parts
    assert set(["driver", "solver", "terminal"]) <= set(parts)


def test_certified_radius_contains_fine_proxy():
    """Not a proof: the level-16 centre is a proxy for the latent path; certified radii must dominate the distance."""
    for seed in range(4):
        br = make(seed)
        fine = br.enclosure(16)
        tt = np.linspace(0, T0, 800)
        ref = fine(tt)
        for n in (6, 9, 12):
            e = br.enclosure(n)
            assert np.max(np.linalg.norm(e(tt) - ref, axis=1)) <= e.radius


def guided_moments(t_end):
    B, c, a = OU2.B, OU2.c, OU2.a
    d = 2

    def rhs(t, y):
        m, P = y[:d], y[d:].reshape(d, d)
        Af = B - np.eye(d) / (T0 - t)
        dm = Af @ m + c + V0 / (T0 - t)
        dP = Af @ P + P @ Af.T + a
        return np.concatenate([dm, dP.ravel()])
    sol = solve_ivp(rhs, [0, t_end], np.concatenate([U0, np.zeros(4)]), rtol=1e-10, atol=1e-12)
    y = sol.y[:, -1]
    return y[:d], y[d:].reshape(d, d)


@pytest.mark.slow
def test_guided_law_midpoint_moments():
    m, P = guided_moments(0.5)
    n = 600
    xs = np.array([make(1000 + s, estimate_solver=False).enclosure(9)(0.5)[0] for s in range(n)])
    se = np.sqrt(np.diag(P) / n)
    assert np.all(np.abs(xs.mean(0) - m) < 4 * se + 0.02)
    assert np.allclose(np.cov(xs.T), P, atol=0.12)


def test_J_intervals_nested_and_contain_reference():
    model = AdditiveModel.from_ou(OU2)
    prop = GuidedProposal(model, U0, V0, T0, np.random.default_rng(5))
    lo_prev, hi_prev = -math.inf, math.inf
    widths = []
    for n in range(1, 12):
        lo, hi = prop.interval(n)
        assert lo >= lo_prev and hi <= hi_prev
        widths.append(hi - lo)
        lo_prev, hi_prev = lo, hi
    assert widths[-1] < widths[0]
    # reference: quadrature of G along the level-18 centre (proxy for the latent path)
    br = prop.bridge
    enc = br.enclosure(18)
    t, X = enc.times[:-1], enc.values[:-1]
    G = -np.einsum("ij,jk,ik->i", X @ OU2.B.T + OU2.c, np.linalg.inv(OU2.a), X - V0) / (T0 - t)
    Jref = np.sum(0.5 * (G[1:] + G[:-1]) * np.diff(t))
    assert lo_prev <= Jref <= hi_prev


@pytest.mark.slow
def test_normaliser_identity_Z_equals_E_Psi():
    """E_Q[Psi] = Z = p(0,u;T,v)/p0~ (Paper 2 Thm 5.4): Monte-Carlo with fine-level J centres."""
    model = AdditiveModel.from_ou(OU2)
    Z = mom.ou_endpoint_data(OU2, T0, U0, V0)["Z"]
    n = 1500
    psi = []
    for s in range(n):
        prop = GuidedProposal(model, U0, V0, T0, np.random.default_rng(9000 + s), estimate_solver=False)
        lo, hi = prop.J_interval_at_level(13)
        psi.append(math.exp(0.5 * (lo + hi)))
    psi = np.array(psi)
    assert abs(psi.mean() - Z) < 4 * psi.std() / math.sqrt(n) + 0.03 * Z


def test_layer_certificate_is_finite_and_dominates_exact_moment():
    """Theorem 8.4 constant vs the exact E_Q[Psi(1+log+ Psi)^2] estimated by Monte Carlo (sanity only)."""
    orc = OUGuidedOracle(OU2, U0, V0, T0)
    cert = orc.layer_certificate()
    assert math.isfinite(cert["M_c"]) and cert["M_c"] > cert["Z_plus"]
    Z = mom.ou_endpoint_data(OU2, T0, U0, V0)["Z"]
    assert cert["Z_plus"] >= Z            # (53): Z <= Z+
