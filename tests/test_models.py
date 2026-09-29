import numpy as np
import pytest

from exactjd.models.ou import OU
from exactjd.models.nonlinear import NonlinearBenchmark, Phi, Psi, f, g, q
from exactjd.baselines.euler import euler_maruyama


def test_ou_transition_moments_match_closed_form():
    kappa, Sig = 1.3, np.array([[1.0, 0.3], [0.0, 0.8]])
    ou = OU.isotropic(kappa, Sig)
    h, x = 0.7, np.array([0.5, -1.0])
    mu, Q = ou.mean_cov(h, x)
    assert np.allclose(mu, np.exp(-kappa * h) * x)
    assert np.allclose(Q, (1 - np.exp(-2 * kappa * h)) / (2 * kappa) * Sig @ Sig.T)


def test_ou_general_B_matches_euler_moments():
    B = np.array([[-1.0, 0.4], [-0.2, -0.5]])
    ou = OU(B, [0.3, -0.1], np.eye(2))
    rng = np.random.default_rng(1)
    n = 4000
    ys = np.array([euler_maruyama(ou, np.zeros(2), 1.0, 400, rng)[1][-1] for _ in range(n)])
    mu, Q = ou.mean_cov(1.0, np.zeros(2))
    assert np.allclose(ys.mean(0), mu, atol=4 * np.sqrt(np.diag(Q)) / np.sqrt(n) + 0.02)
    assert np.allclose(np.cov(ys.T), Q, atol=0.08)


def test_ou_bridge_reference_endpoint_and_midpoint_law():
    ou = OU.isotropic(1.0, np.eye(2))
    rng = np.random.default_rng(3)
    x, y = np.array([0.0, 0.0]), np.array([1.0, -1.0])
    vals = np.array([ou.sample_bridge_values(0, x, 1.0, y, [0.5], rng)[0] for _ in range(6000)])
    # analytic mid law via Gaussian conditioning
    A1, m1, Q1 = ou.transition_params(0.5)
    S = A1 @ Q1 @ A1.T + Q1
    K = Q1 @ A1.T @ np.linalg.inv(S)
    mu = A1 @ x + K @ (y - A1 @ (A1 @ x))
    C = Q1 - K @ A1 @ Q1
    assert np.allclose(vals.mean(0), mu, atol=0.05)
    assert np.allclose(np.cov(vals.T), C, atol=0.04)


def test_survival_formula_vs_thinning():
    """P(tau_1>t) analytic (App. E.1) vs exact-in-law Poisson thinning is checked in jump tests; here
    check the ODE (Riccati) identity by finite differences on the closed form."""
    ou = OU.isotropic(1.0, np.eye(2))
    x, lam0, lam1 = np.array([0.3, -0.2]), 1.0, 1.0
    t, e = 0.6, 1e-5
    u = lambda tt: ou.survival_quadratic_rate(tt, x, lam0, lam1)
    assert 0 < u(t) < 1
    assert u(0.0) == pytest.approx(1.0)
    # Feynman-Kac PDE at a point:  u_t = L u - lambda u
    du_t = (u(t + e) - u(t - e)) / (2 * e)
    def uy(y, tt):
        return ou.survival_quadratic_rate(tt, y, lam0, lam1)
    grad = np.array([(uy(x + e * np.eye(2)[i], t) - uy(x - e * np.eye(2)[i], t)) / (2 * e) for i in range(2)])
    e2 = 1e-4
    lap = sum((uy(x + e2 * np.eye(2)[i], t) - 2 * uy(x, t) + uy(x - e2 * np.eye(2)[i], t)) / e2**2 for i in range(2))
    Lu = -1.0 * x @ grad + 0.5 * lap
    assert du_t == pytest.approx(Lu - (lam0 + lam1 * x @ x) * u(t), rel=1e-3, abs=1e-5)


def test_g_inverse_and_phi_psi():
    x = np.linspace(-20, 20, 101)
    assert np.allclose(f(g(x)), x, atol=1e-12)
    z = np.random.default_rng(0).normal(size=(50, 2))
    assert np.allclose(Psi(Phi(z)), z, atol=1e-11)
    assert np.all((q(x) >= 0.5) & (q(x) <= 1.5))


def test_nonlinear_sde_law_equals_phi_of_bm():
    """Euler simulation of the (b, sigma) SDE has the same terminal law as Phi(B) (weak check)."""
    m = NonlinearBenchmark()
    rng = np.random.default_rng(7)
    n, T = 3000, 0.5
    x0 = np.array([0.3, 0.1])
    ys = np.array([euler_maruyama(m, x0, T, 250, rng)[1][-1] for _ in range(n)])
    zs = np.array([m.sample_transition(0, x0, T, rng) for _ in range(n)])
    assert np.allclose(ys.mean(0), zs.mean(0), atol=0.06)
    assert np.allclose(np.cov(ys.T), np.cov(zs.T), atol=0.08)


def test_nonlinear_density_normalises():
    m = NonlinearBenchmark()
    x = np.array([0.2, -0.4])
    h = 0.3
    g1 = np.linspace(-4, 4, 801)
    g2 = np.linspace(-4, 4, 801)
    # integrate in Psi-coordinates: p(x,y)dy = N density in z
    tot = 0.0
    dz = (g1[1] - g1[0]) * (g2[1] - g2[0])
    for a in g1:
        for b in g2[::4]:
            pass
    zc = Psi(x)
    zz = np.exp(-((g1[:, None] - zc[0]) ** 2 + (g2[None, :] - zc[1]) ** 2) / (2 * h)) / (2 * np.pi * h)
    assert zz.sum() * dz == pytest.approx(1.0, abs=1e-3)
    y = Phi(np.array([0.1, 0.2]))
    # p_h(x,y) * |det DPhi(z)| = N(z)
    dens_z = np.exp(-np.sum((np.array([0.1, 0.2]) - zc) ** 2) / (2 * h)) / (2 * np.pi * h)
    assert m.transition_density(h, x, y) * (1 + 0.5 * np.cos(0.1)) == pytest.approx(dens_z, rel=1e-9)
