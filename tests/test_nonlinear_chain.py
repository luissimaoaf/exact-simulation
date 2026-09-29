import math

import numpy as np
import pytest

from exactjd.correction.abstract import retrospective_select
from exactjd.correction.nonlinear import NonlinearGuidedOracle
from exactjd.jump import skeleton as sk
from exactjd.jump.records import (OUBridgeFactory, P2BridgeFactory, TransformedBridgeFactory, _TruncatedEnvelope)
from exactjd.models import nonlinear as nl
from exactjd.models.jump import QuadraticRateJump, TransformedQuadraticRateJump
from exactjd.models.ou import OU
from exactjd.rng import Budget, BudgetExhausted


def test_transformed_bridge_record_certificates():
    rng = np.random.default_rng(0)
    for _ in range(15):
        x = nl.Phi(rng.normal(size=2) * 0.5)
        y = nl.Phi(rng.normal(size=2) * 0.5)
        rec = TransformedBridgeFactory().new_bridge(0.0, x, 0.7, y, rng)
        e = rec.enclosure(14)
        assert np.allclose(e.values[0], x) and np.allclose(e.values[-1], y)
        ref = float(np.max(np.linalg.norm(e.values, axis=1)))
        lo, hi = rec.max_norm_interval(9)
        assert lo <= ref + e.radius and hi >= ref - 1e-9


@pytest.mark.slow
def test_nonlinear_direct_route_first_event_survival():
    """Direct transformed route: the model in x-coordinates equals Phi(BM); analytic survival with kappa=0 in z-coordinates."""
    ou0 = OU.isotropic(0.0, np.eye(2))
    jN = TransformedQuadraticRateJump(1.0, 1.0, 0.5, 0.2)

    class NL:
        d = m = 2

        def sample_transition(self, s, x, t, rng):
            return nl.Phi(nl.Psi(np.asarray(x)) + math.sqrt(t - s) * rng.standard_normal(2))

    x0 = nl.Phi(np.array([0.3, 0.2]))
    rng = np.random.default_rng(3)
    n = 500
    tau = []
    for _ in range(n):
        r = sk.next_event(NL(), jN, TransformedBridgeFactory(), 0.0, x0, 1.0, rng, keep_records=False)
        tau.append(r.tau if r.kind == "event" else math.inf)
    tau = np.array(tau)
    for t in (0.5, 1.0):
        ref = ou0.survival_quadratic_rate(t, nl.Psi(x0), 1.0, 1.0)
        assert abs((tau > t).mean() - ref) < 4 * math.sqrt(ref * (1 - ref) / n) + 1e-3


def test_kappa_zero_ou_bridge_record_matches_brownian():
    ou0 = OU.isotropic(0.0, np.eye(2))
    rec = OUBridgeFactory(ou0).new_bridge(0.0, np.zeros(2), 1.0, np.array([0.5, -0.5]), np.random.default_rng(1))
    e = rec.refine_to(0.3)
    assert e.radius <= 0.3 and np.allclose(e.values[-1], [0.5, -0.5])


def test_p2_bridge_record_in_jump_skeleton_diagnostic_mode_smoke():
    """One chain step: Paper 1 (additive guided bridge) -> Paper 2 (truncated race, diagnostic radii) -> Paper 3 record."""
    ou = OU.isotropic(1.0, np.eye(2))
    fac = P2BridgeFactory(ou, mode="truncated", race_kmax=3, max_factory_coins=3000,
                          budget=Budget(max_seconds=100), certification="diagnostic")
    done = False
    for seed in range(4):
        try:
            rec = fac.new_bridge(0.0, np.array([0.3, 0.0]), 0.1, np.array([0.25, 0.1]), np.random.default_rng(seed))
            e = rec.refine_to(0.5)
            lo, hi = rec.max_norm_interval(3)
            assert 0.0 <= lo <= hi
            assert np.allclose(e.values[0], [0.3, 0.0], atol=1e-9) and np.allclose(e.values[-1], [0.25, 0.1], atol=1e-9)
            assert fac.tv_bounds and fac.tv_bounds[0] < 0.1          # Chebyshev truncation bound is reported
            done = True
            break
        except BudgetExhausted:
            continue
    assert done


def test_nonlinear_guided_correction_runs_and_is_flagged():
    orc = NonlinearGuidedOracle(np.array([0.0, 0.0]), np.array([0.3, 0.1]), 0.5)
    rec = orc.new_proposal(np.random.default_rng(0))
    lo, hi = rec.interval(1)
    assert lo < hi
    assert rec.status.startswith("diagnostic")
    lo2, hi2 = rec.interval(3)
    assert lo2 >= lo and hi2 <= hi and hi2 - lo2 < hi - lo
    sel = retrospective_select(orc, _TruncatedEnvelope(4.0, -8, 3), np.random.default_rng(1), budget=Budget(max_seconds=120))
    assert sel.diagnostics.completed
