import math

import numpy as np
import pytest

from exactjd.brownian import tailfirst as tf
from exactjd.brownian.record import BrownianRecord
from exactjd.brownian.levyarea import BrownianAreaRecord
from exactjd.brownian import rough


def test_last_failure_distribution():
    rng = np.random.default_rng(0)
    pi = np.array([0.5, 0.3, 0.1, 0.02])
    samples = np.array([tf.sample_last_failure(pi, rng) for _ in range(40000)])
    # exact law: P(K<=k)=prod_{j>k}(1-pi_j)
    G = [np.prod(1 - pi[k:]) for k in range(len(pi) + 1)]
    emp = np.array([(samples <= k).mean() for k in range(len(pi) + 1)])
    assert np.allclose(emp, G, atol=0.01)


def test_constrained_vector_exceed_law():
    rng = np.random.default_rng(1)
    t = np.array([1.0, 1.5, 0.5])
    n = 30000
    z = np.array([tf.constrained_vector(rng, t, True) for _ in range(n)])
    assert np.all((np.abs(z) > t).any(axis=1))
    # conditional law of z_1 given exceedance: P(|z_1|>1 | exceed)
    p = tf.tail_prob(t)
    pf = 1 - np.prod(1 - p)
    assert (np.abs(z[:, 0]) > 1.0).mean() == pytest.approx(p[0] / pf, abs=0.01)


def test_record_marginals_are_brownian():
    """Mixture over K* must reproduce the unconditioned law: check variance of nodes and increments."""
    n, T = 3000, 1.0
    vals = []
    for s in range(n):
        r = BrownianRecord(T, 2, np.random.default_rng(s))
        nd = r.nodes(4)
        vals.append([nd[8, 0], nd[4, 0], nd[12, 1], nd[16 if len(nd) > 16 else -1, 0]])
    vals = np.array(vals)
    assert vals[:, 0].var() == pytest.approx(0.5, rel=0.08)
    assert np.cov(vals[:, 0] - vals[:, 1], vals[:, 1])[0, 1] == pytest.approx(0.0, abs=0.03)
    assert vals[:, 3].var() == pytest.approx(1.0, rel=0.08)


def test_refinement_is_nested():
    r = BrownianRecord(1.0, 2, np.random.default_rng(5))
    a = r.nodes(5).copy()
    b = r.nodes(9)
    assert np.array_equal(a, b[::16])
    # radius decreases
    assert r.radius(3) > r.radius(6) > r.radius(9) > 0


def test_radius_contains_fine_path():
    """Certified radius at level n must dominate distance to (proxy for) the true path, level n+12 node values."""
    for s in range(30):
        r = BrownianRecord(1.0, 2, np.random.default_rng(100 + s))
        fine = r.nodes(14)
        for n in (2, 4, 6):
            poly = r.polygon(n)
            tt = np.linspace(0, 1, len(fine))
            err = np.max(np.linalg.norm(fine - poly(tt), axis=1))
            assert err <= r.radius(n) + 1e-12


def test_holder_constant_dominates_increments():
    r = BrownianRecord(1.0, 2, np.random.default_rng(11))
    fine = r.nodes(12)
    H = r.holder_constant(0.45)
    idx = np.random.default_rng(0).integers(0, len(fine), size=(4000, 2))
    for i, j in idx:
        if i != j:
            l = abs(i - j) / (len(fine) - 1)
            assert np.linalg.norm(fine[i] - fine[j]) <= H * l**0.45 + 1e-12


def test_area_generator_covariance_and_chen():
    n, T, lev = 2500, 1.0, 9
    areas, incs = [], []
    for s in range(n):
        r = BrownianAreaRecord(T, np.random.default_rng(1000 + s))
        nd = r.nodes(lev)
        cum = rough.cumulative_area(nd)
        areas.append(rough.area(nd, 0, len(nd) - 1, cum))
        incs.append(nd[-1] - nd[0])
    areas, incs = np.array(areas), np.array(incs)
    assert abs(areas.mean()) < 4 * areas.std() / np.sqrt(n)     # 4 standard errors
    assert areas.var() == pytest.approx(T**2 / 4 * (1 - 2.0**-lev), rel=0.1)   # Var(Levy area)=T^2/4
    assert incs.var(axis=0) == pytest.approx([T, T], rel=0.1)
    assert np.corrcoef(areas, incs[:, 0])[0, 1] == pytest.approx(0.0, abs=0.06)


def test_chen_identity_exact():
    r = BrownianAreaRecord(1.0, np.random.default_rng(3))
    nd = r.nodes(8)
    assert rough.chen_defect(nd, 3, 40, 200) < 1e-13
    D, S = rough.lift(nd, 5, 100)
    assert np.allclose(0.5 * (S + S.T), 0.5 * np.outer(D, D))  # geometric: Sym = 1/2 D(x)D


def test_residual_identity_matches_polygon_areas():
    """sum_k S_k over levels n<k<=m equals the direct polygon-area difference (Chen split derivation)."""
    r = BrownianAreaRecord(1.0, np.random.default_rng(9))
    n, m = 4, 9
    ndm, ndn = r.nodes(m), r.nodes(n)
    cm, cn = rough.cumulative_area(ndm), rough.cumulative_area(ndn)
    for (i, j) in [(0, 16), (2, 7), (5, 6), (3, 12)]:
        direct = rough.area(ndm, i * 2 ** (m - n), j * 2 ** (m - n), cm) - rough.area(ndn, i, j, cn)
        assert direct == pytest.approx(r.residual(n, m, i, j), abs=1e-12)


def test_area_certificates_dominate_and_decay():
    r = BrownianAreaRecord(1.0, np.random.default_rng(21))
    alpha, abar, beta = 0.34, 0.49, 0.6
    Es = [r.constants(alpha, abar, beta, n)["E"] for n in (4, 8, 12, 20, 40)]
    assert all(Es[i] > Es[i + 1] for i in range(len(Es) - 1))
    # empirical residual dominated by aggregate certificate G
    n, m = 6, 12
    c = r.constants(alpha, abar, beta, n)
    ndm = r.nodes(m)
    cm = rough.cumulative_area(ndm)
    hn = 2.0**-n
    worst = 0.0
    for i in range(0, 64, 3):
        for j in range(i + 1, 65, 5):
            res = math.sqrt(2) * abs(r.residual(n, m, i, j))
            worst = max(worst, res / (((j - i) * hn) ** beta * hn ** (2 * abar - beta)))
    assert worst <= c["G"] * 1.0 + 1e-12


def test_rho_bound_soundness_and_size_report():
    """Certified rho_alpha bound dominates the (numerical) distance between fine and coarse lifts."""
    r = BrownianAreaRecord(1.0, np.random.default_rng(4))
    alpha, abar, beta = 0.34, 0.49, 0.6
    n, m = 5, 11
    est = rough.rho_alpha_on_grid(r.nodes(m), n, m, 1.0, alpha)
    cert = r.constants(alpha, abar, beta, n)["E"]
    assert est <= cert
    assert cert > 1.0   # honest: certificate is (very) conservative at practical levels
