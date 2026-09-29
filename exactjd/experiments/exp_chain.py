"""End-to-end chains.

  1. OU chain:  Paper 1 (additive-noise guided bridge)  ->  Paper 2 (retrospective correction, exact selection)  ->
     Paper 3 (jump skeleton with bridges supplied by Paper 2).  Two variants:
       * certified-truncated : rigorous (ideal arithmetic) enclosures, SUPPORT-TRUNCATED race (approximate; TV bound reported);
       * diagnostic          : calibrated UNCERTIFIED radii, same truncated race (approximate + uncertified).
     Every requested run is logged; censored runs are counted (refinement-depth / factory-scale / time limits).
  2. Nonlinear benchmark, DIRECT transformed route (independent reference, not Papers 1-2), cross-checked against the
     analytic first-event survival of the Brownian-coordinate model.
  3. Nonlinear benchmark, guided (Paper 1 rough route) -> correction (Paper 2) in DIAGNOSTIC mode: law of the corrected
     bridge midpoint vs the direct reference, and vs the UNcorrected guided proposal (to show the correction matters).

    python -m exactjd.experiments.exp_chain [--full]
Outputs: results/chain/report.md, runs.jsonl
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ..correction import bridge as cb
from ..correction.abstract import LogEnvelope, retrospective_select
from ..correction.nonlinear import NonlinearGuidedOracle
from ..jump import skeleton as sk
from ..jump.records import OUBridgeFactory, P2BridgeFactory, TransformedBridgeFactory, _TruncatedEnvelope
from ..models import nonlinear as nl
from ..models.jump import QuadraticRateJump, TransformedQuadraticRateJump
from ..models.ou import OU
from ..rng import Budget
from .exp_AB_reference import md_table
from .runner import ExperimentLog, run_experiment, skeleton_run, summarize


def main(full: bool = False, out: str = "results/chain") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = ExperimentLog(out / "runs.jsonl")
    rep = []
    ou = OU.isotropic(1.0, np.eye(2))
    jump = QuadraticRateJump(2, lam0=1.0, lam1=1.0, rho=0.5, jump=0.2)
    x0, T = np.array([0.3, 0.0]), 0.1

    # ---------------------------------------------------------------- 1. OU chain skeleton runs
    for cert in ("diagnostic", "certified"):
        cfg = f"ou_chain_{cert}"
        fac = lambda b, cert=cert: P2BridgeFactory(ou, mode="truncated", race_kmax=3, max_factory_coins=5000, budget=b,
                                                    certification=cert)

        def fn(rng, budget, ctx, fac=fac):
            return skeleton_run(ou, jump, fac, x0, T, rng, budget, ctx, refine_eps=0.3)
        rows = run_experiment(cfg, 6 if not full else 20, fn, log, seed=21, max_seconds=(400 if cert == "diagnostic" else 200),
                              verbose=False)
        s = summarize(rows)
        comp = [r["result"] for r in rows if r["status"] == "completed"]
        rep.append(f"## OU chain ({cert} radii, truncated race kmax=3), T={T}, quadratic intensity\n\n"
                   f"requested {s['requested']}, completed {s['completed']}, censored {s['censored']}, "
                   f"wall median {s.get('wall_q50', 0):.0f}s / max {s.get('wall_max', 0):.0f}s.  Completed skeletons: "
                   f"events {[c['n_events'] for c in comp]}, refined radius {[round(c.get('refined_radius', float('nan')), 3) for c in comp]}.\n"
                   f"Censoring reasons: {sorted({r.get('reason', '')[:70] for r in rows if r['status'] == 'censored'})}\n")

    # ---------------------------------------------------------------- 2. nonlinear DIRECT route vs analytic survival
    ou0 = OU.isotropic(0.0, np.eye(2))                    # BM in Brownian coordinates: analytic survival with kappa=0
    jN = TransformedQuadraticRateJump(1.0, 1.0, 0.5, 0.2)
    class NL:
        d = m = 2
        def sample_transition(self, s, x, t, rng):
            return nl.Phi(nl.Psi(np.asarray(x)) + math.sqrt(t - s) * rng.standard_normal(2))
    xN = nl.Phi(np.array([0.3, 0.2]))
    n = 800 if not full else 3000
    rng = np.random.default_rng(31)
    tau = []
    fac = TransformedBridgeFactory()
    for _ in range(n):
        r = sk.next_event(NL(), jN, fac, 0.0, xN, 1.0, rng, keep_records=False)
        tau.append(r.tau if r.kind == "event" else math.inf)
    tau = np.array(tau)
    rows = []
    for t in (0.25, 0.5, 1.0):
        ref = ou0.survival_quadratic_rate(t, nl.Psi(xN), 1.0, 1.0)
        emp = (tau > t).mean()
        se = math.sqrt(ref * (1 - ref) / n)
        rows.append([t, f"{emp:.3f}", f"{ref:.3f}", f"{(emp-ref)/se:+.2f}"])
    rep.append(f"\n## Nonlinear benchmark, DIRECT transformed route (not Papers 1-2): first-event survival, N={n}\n\n"
               + md_table(["t", "empirical", "analytic (Brownian coords, kappa=0)", "z-score"], rows))

    # ---------------------------------------------------------------- 3. nonlinear guided -> correction (diagnostic)
    u, v, Tb = np.array([0.0, 0.0]), np.array([0.3, 0.1]), 0.5
    nb = 60 if not full else 300
    log2 = ExperimentLog(out / "nonlinear_p2.jsonl")

    def fn3(rng, budget, ctx):
        orc = NonlinearGuidedOracle(u, v, Tb, budget)
        env = _TruncatedEnvelope(4.0, -8, 3)
        sel = retrospective_select(orc, env, rng, budget=budget)
        br = sel.proposal.bridge
        ts, xs = br.center(9, 6, 2)
        return dict(mid=[float(np.interp(0.25, ts, xs[:, j])) for j in range(2)], trials=sel.diagnostics.race_trials,
                    coins=sel.diagnostics.membership_coins)
    rows_ = run_experiment("nonlinear_guided_corrected", nb, fn3, log2, seed=41, max_seconds=180, verbose=False)
    comp = [r["result"]["mid"] for r in rows_ if r["status"] == "completed"]
    # references: direct transformed bridge midpoint, and UNcorrected guided proposal
    rng = np.random.default_rng(5)
    nl_model = nl.NonlinearBenchmark()
    ref = np.array([nl_model.sample_bridge_values(0.0, u, Tb, v, [0.25], rng)[0] for _ in range(2000)])
    from ..bridges.rough_guided import RoughGuidedBridge
    from ..bridges.roughmodels import NonlinearRoughModel
    unc = []
    for s in range(nb if not full else 300):
        br = RoughGuidedBridge(NonlinearRoughModel(), u, v, Tb, np.random.default_rng(700 + s))
        ts, xs = br.center(9, 6, 2)
        unc.append([float(np.interp(0.25, ts, xs[:, j])) for j in range(2)])
    unc = np.array(unc)
    comp = np.array(comp) if comp else np.zeros((0, 2))
    def line(name, a):
        if len(a) < 3:
            return [name, len(a), "-", "-"]
        return [name, len(a), f"{a[:, 0].mean():+.3f} ± {a[:, 0].std()/math.sqrt(len(a)):.3f}, {a[:, 1].mean():+.3f} ± {a[:, 1].std()/math.sqrt(len(a)):.3f}",
                f"{a[:, 0].std():.3f}, {a[:, 1].std():.3f}"]
    s3 = summarize(rows_)
    rep.append(f"\n## Nonlinear benchmark: guided (Paper 1, rough route) -> correction (Paper 2), DIAGNOSTIC (uncertified J, truncated race)\n\n"
               f"requested {s3['requested']}, completed {s3['completed']}, censored {s3['censored']}; wall median {s3.get('wall_q50', 0):.1f}s.\n\n"
               + md_table(["law of X_{T/2}", "N", "mean ± se (x1, x2)", "sd (x1, x2)"],
                          [line("direct transformed reference", ref), line("guided proposal (uncorrected)", unc),
                           line("guided -> Paper 2 correction (diagnostic)", comp)]))
    (out / "report.md").write_text("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
