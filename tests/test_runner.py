import math
import pickle

import numpy as np
import pytest

from exactjd.experiments import runner as R
from exactjd.models.ou import OU
from exactjd.models.jump import QuadraticRateJump
from exactjd.jump.records import OUBridgeFactory
from exactjd.rng import Budget, BudgetExhausted, StreamFactory


def test_streams_are_independent_and_reproducible():
    f = StreamFactory(7)
    a1, a2 = f.generator("cfg", 0).random(4), f.generator("cfg", 0).random(4)
    b = f.generator("cfg", 1).random(4)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


def test_censored_runs_are_recorded_never_dropped(tmp_path):
    log = R.ExperimentLog(tmp_path / "log.jsonl")

    def fn(rng, budget, ctx):
        if ctx["run"] % 3 == 0:
            raise BudgetExhausted("work limit")
        return dict(x=float(rng.random()))

    rows = R.run_experiment("cfg", 9, fn, log, seed=1, verbose=False)
    s = R.summarize(rows)
    assert s["requested"] == 9 and s["censored"] == 3 and s["completed"] == 6
    assert s["completion_fraction"] == pytest.approx(6 / 9)
    n_lines = len(log.rows())
    R.run_experiment("cfg", 9, fn, log, seed=1, verbose=False)      # resume: nothing re-run, nothing dropped
    assert len(log.rows()) == n_lines


def test_interruption_is_detected_and_recorded_on_resume(tmp_path):
    log = R.ExperimentLog(tmp_path / "log.jsonl")
    log.append(dict(config="c", run=0, status="started", seed=0))       # process died here
    rows = R.run_experiment("c", 2, lambda rng, b, ctx: dict(ok=1), log, seed=0, verbose=False)
    statuses = [r["status"] for r in log.rows() if r["config"] == "c"]
    assert "interrupted" in statuses
    assert all(r["status"] == "completed" for r in rows)


def test_skeleton_checkpoint_resume_identity(tmp_path):
    ou = OU.isotropic(1.0, np.eye(2))
    jump = QuadraticRateJump(2, 3.0, 1.0, 0.5)
    fac = lambda b: OUBridgeFactory(ou, b)
    x0 = np.array([0.5, 0.5])
    full = R.skeleton_run(ou, jump, fac, x0, 1.0, np.random.default_rng(5), Budget(), dict(ckpt_path=None))
    ck = tmp_path / "ck.pkl"
    ncall = {"n": 0}
    from exactjd.jump.skeleton import simulate_skeleton

    def dying_checkpoint(state):
        ncall["n"] += 1
        with open(ck, "wb") as f:
            pickle.dump(state, f)
        if ncall["n"] == 2:
            raise KeyboardInterrupt

    if full["n_events"] >= 1:
        with pytest.raises(KeyboardInterrupt):
            simulate_skeleton(ou, jump, fac(Budget()), x0, 1.0, np.random.default_rng(5), checkpoint=dying_checkpoint)
        resumed = R.skeleton_run(ou, jump, fac, x0, 1.0, np.random.default_rng(123), Budget(), dict(ckpt_path=ck))
        assert resumed["n_events"] == full["n_events"]
        assert resumed["taus"] == full["taus"]
        assert resumed["terminal"] == full["terminal"]
