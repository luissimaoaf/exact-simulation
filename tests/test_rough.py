import math

import numpy as np
import pytest

from exactjd.bridges import cpverify as cp
from exactjd.bridges.coeffbounds import DerivTensorBounds
from exactjd.bridges.roughmodels import NonlinearRoughModel, HestonRoughModel
from exactjd.bridges.rough_guided import RoughGuidedBridge
from exactjd.baselines.euler import euler_maruyama
from exactjd.brownian import rough
from exactjd.models import nonlinear as nl
from exactjd.rng import Budget, BudgetExhausted

import sympy as sp


def test_interval_bounds_dominate_sampled_derivatives():
    x, y = sp.symbols("x y", real=True)
    fields = [[sp.sin(x) * sp.cos(y), sp.exp(-x**2)], [x * y, sp.sqrt(1 + x**2)]]
    dtb = DerivTensorBounds(fields, [x, y], rmax=3)
    lo, hi = [-1.0, -0.5], [0.8, 1.2]
    rng = np.random.default_rng(0)
    for r in range(0, 4):
        B = dtb.bound(r, lo, hi, depth=3)
        for _ in range(60):
            pt = [rng.uniform(l, h) for l, h in zip(lo, hi)]
            tot = 0.0
            for fn in dtb.fns[r]:
                a_, b_ = fn.enclosure(pt, pt)
                tot += (0.5 * (a_ + b_)) ** 2
            assert math.sqrt(tot) <= B * (1 + 1e-9)


def test_nonlinear_model_coefficients_match_paper():
    m = NonlinearRoughModel().bind([0.4, 0.2], 1.0)
    for x in ([0.1, 0.3], [-1.3, 2.0]):
        x = np.array(x)
        q = nl.q(x[0])
        assert np.allclose(m.a(x), np.diag([q**2, 1.0]))
        assert np.allclose(m.F(x), [q / 2, 0.0], atol=1e-12)                      # F=(q/2,0)
        assert np.allclose(m.V(x) @ m.V(x).T, m.a(x))
        # Ito drift recovered from F: b=(-sin(g(x1))/4, 0)
        assert np.allclose(m.drift_b(x), [-0.25 * np.sin(nl.g(x[0])), 0.0], atol=1e-5)
    K = m.V_bounds([0.0, 0.0], 8.0, range(0, 6))
    assert all(np.isfinite(list(K.values())))
    assert K[0] > 1.0                        # sup|V|_HS >= |V| at a point


def test_heston_conversion_and_fifth_derivative_growth():
    kap, th, xi, rho = 2.0, 0.04, 0.3, -0.5
    m = HestonRoughModel(kap, th, xi, rho, r=0.03).bind([0.0, 0.05], 1.0)
    x = np.array([0.1, 0.06])
    assert np.allclose(m.drift_b(x), [0.03 - x[1] / 2, kap * (th - x[1])], atol=1e-5)   # Ito drift of Heston
    assert np.allclose(m.a(x), x[1] * np.array([[1, xi * rho], [xi * rho, xi**2]]))
    k1 = m.V_bounds([0.0, 0.2], 0.05, [5])[5]      # z in [0.15,0.25]
    k2 = m.V_bounds([0.0, 0.1], 0.025, [5])[5]     # z in [0.075,0.125]
    assert k2 / k1 > 8.0                           # ~ z^{-9/2}: derivative bounds blow up at the boundary (paper §10)


def _toy_driver(T, N, eps):
    t = np.linspace(0, T, N + 1)
    U0 = np.stack([t, 0.5 * np.sin(3 * t)], axis=1)
    U1 = U0 + eps * np.stack([np.sin(40 * t), np.cos(25 * t) - 1], axis=1) * np.sqrt(T)
    return t, U0, U1


def _solve(model, t, U, x0):
    x = np.array(x0, float)
    out = [x.copy()]
    for i in range(len(t) - 1):
        dU = (U[i + 1] - U[i]) / (t[i + 1] - t[i])
        dt = t[i + 1] - t[i]
        k1 = model.V(x) @ dU
        k2 = model.V(x + dt / 2 * k1) @ dU
        k3 = model.V(x + dt / 2 * k2) @ dU
        k4 = model.V(x + dt * k3) @ dU
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.array(out)


def test_cpverify_radius_dominates_true_two_driver_difference():
    """Soundness of the block recursion on a case where the certificate is attainable (small T, small driver ball)."""
    model = NonlinearRoughModel().bind([0.4, 0.2], 1.0)
    T, N, eps, alpha = 5e-6, 100, 0.05, 0.4
    t, U0, U1 = _toy_driver(T, N, eps)
    X0, X1 = _solve(model, t, U0, [0.3, 0.1]), _solve(model, t, U1, [0.3, 0.1])
    true_diff = float(np.max(np.linalg.norm(X0 - X1, axis=1)))
    # driver quantities from the polygonal (canonical) lifts on the fine grid
    def norms(U):
        best = 0.0
        cum = rough.cumulative_area(U)
        for a in range(0, N, 5):
            for b in range(a + 1, N + 1, 5):
                D, S = rough.lift(U, a, b, cum)
                L = t[b] - t[a]
                best = max(best, np.linalg.norm(D) / L**alpha, math.sqrt(np.linalg.norm(S)) / L**alpha)
        return best
    def rho(Ua, Ub):
        best = 0.0
        ca, cb = rough.cumulative_area(Ua), rough.cumulative_area(Ub)
        for a in range(0, N, 5):
            for b in range(a + 1, N + 1, 5):
                Da, Sa = rough.lift(Ua, a, b, ca)
                Db, Sb = rough.lift(Ub, a, b, cb)
                L = t[b] - t[a]
                best = max(best, np.linalg.norm(Da - Db) / L**alpha, math.sqrt(np.linalg.norm(Sa - Sb)) / L**alpha)
        return best
    R = 1.5 * max(norms(U0), norms(U1), 1.0)
    delta = 1.5 * rho(U0, U1)
    K = model.V_bounds([0.3, 0.1], 3.0, range(0, 4))
    bounds = cp.Bounds((K[0], K[1], K[2], K[3]), (0.0, 0.0, 0.0), 0.0, 0.0)
    r_in, a, K_end = 0.0, 0, N
    length = N
    blocks = 0
    while a < K_end:
        length = min(length, K_end - a)
        while True:
            b = a + length
            ell = t[b] - t[a]
            res = cp.verify_block(lambda rad: bounds, ell, alpha, R * ell**alpha, R**2 * ell ** (2 * alpha), r_in, delta)
            if res.ok:
                break
            assert length > 1, res.reason
            length //= 2
        r_in, a, blocks = res.radius, b, blocks + 1
    assert r_in >= true_diff * (1 - 1e-9)
    assert r_in < 1e4 * max(true_diff, 1e-12)      # conservative but finite
    assert true_diff > 0


def test_rough_guided_center_matches_guided_law_ito_strato():
    """Guided law check (covariance-matched guide): centre at level 9 vs Euler simulation of the Ito guide SDE (baseline)."""
    model = NonlinearRoughModel().bind([0.6, 0.2], 1.0)
    T, u, v = 1.0, np.array([0.0, 0.0]), np.array([0.6, 0.2])
    n = 250
    xs = []
    for s in range(n):
        br = RoughGuidedBridge(NonlinearRoughModel(), u, v, T, np.random.default_rng(300 + s))
        ts, X = br.center(9, 5, 2)
        xs.append(np.array([np.interp(0.5, ts, X[:, j]) for j in range(2)]))
    xs = np.array(xs)
    # Ito guide: dX=[b + a a_T^{-1}(v-X)/(T-t)]dt + sigma dW   (Euler, small steps, stop at T-0.02)
    class G:
        d = m = 2
        def drift(self, t, x):
            return model.drift_b(x) + model.a(x) @ model.S @ (v - x) / (T - t)
        def sigma(self, t, x):
            return model.V(x)
    rng = np.random.default_rng(1)
    ys = []
    for _ in range(n):
        tt, path = euler_maruyama(G(), u, 0.98, 490, rng)
        ys.append(np.array([np.interp(0.5, tt, path[:, j]) for j in range(2)]))
    ys = np.array(ys)
    se = np.sqrt((xs.var(0) + ys.var(0)) / n)
    assert np.all(np.abs(xs.mean(0) - ys.mean(0)) < 4.5 * se + 0.03)
    tol = 4.0 * math.sqrt(2.0 / n) * np.maximum(np.diag(np.cov(xs.T)), np.diag(np.cov(ys.T)))
    assert np.all(np.abs(np.diag(np.cov(xs.T)) - np.diag(np.cov(ys.T))) < tol + 0.02)
    assert abs(np.cov(xs.T)[0, 1]) < 0.15 and abs(np.cov(ys.T)[0, 1]) < 0.15


def test_certified_pipeline_reports_infeasibility_honestly():
    """With the paper's constants the certified pipeline either yields a huge radius or raises an explicit failure --
    never a silent approximation.  (Numbers are reported by experiments/exp_C_guided.py.)"""
    br = RoughGuidedBridge(NonlinearRoughModel(), [0.0, 0.0], [0.6, 0.2], 1.0, np.random.default_rng(3), budget=Budget())
    d = br.driver(8)
    assert d["delta"] > 1.0 and d["R"] > 1.0        # certificate scale at practical levels
    with pytest.raises((BudgetExhausted, cp.CertificateFailure)):
        br.certified_enclosure(8, max_blocks=200)


def test_diagnostic_enclosure_is_flagged_uncertified():
    br = RoughGuidedBridge(NonlinearRoughModel(), [0.0, 0.0], [0.6, 0.2], 1.0, np.random.default_rng(4))
    e = br.diagnostic_enclosure(7)
    assert e.status.startswith("diagnostic") and "UNCERTIFIED" in e.status
    assert np.allclose(e(1.0), [0.6, 0.2]) and np.allclose(e(0.0), [0.0, 0.0])


def test_open_domain_shell_search_heston_same_driver_never_resampled():
    from exactjd.bridges.localization import BoxExhaustion, LocalizedGuidedBridge
    kap, th, xi, rho = 2.0, 0.04, 0.3, -0.5
    model = HestonRoughModel(kap, th, xi, rho)
    u, v, T = np.array([0.0, 0.04]), np.array([0.05, 0.05]), 1.0
    exh = BoxExhaustion.heston(0.0, 0.04, n=10)
    found = 0
    for seed in range(6):
        br = LocalizedGuidedBridge(model, exh, u, v, T, np.random.default_rng(seed))
        nodes_before = br.record.nodes(6).copy()
        try:
            res = br.shell_search(levels=range(6, 9), diagnostic_radius=0.002)
        except BudgetExhausted:
            continue
        found += 1
        assert res["status"].startswith("diagnostic")
        assert np.array_equal(br.record.nodes(6), nodes_before)      # driver retained, not resampled by the search
        lo, hi = res["center_range"]
        assert lo[1] > 0                                              # centre stays in D = R x (0,inf)
    assert found >= 1
