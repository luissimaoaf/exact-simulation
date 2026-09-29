"""Chain Paper 1 -> 2 -> 3 with a higher intensity so that completed skeletons contain jump events.

    python -m exactjd.experiments.exp_chain_events [--runs 24]

Same components as exp_chain (diagnostic radii, support-truncated race, TV bound reported) on the OU model, T=0.1,
lambda(x)=6+3|x|^2.  All requested runs are logged (results/chain/events_runs.jsonl); censored runs are counted.
The output shows exact event data plus the ε-refined segments built from Paper-2 bridge records.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..jump.records import P2BridgeFactory
from ..models.jump import QuadraticRateJump
from ..models.ou import OU
from .runner import ExperimentLog, run_experiment, skeleton_run, summarize


def main(runs: int = 24, out: str = "results/chain") -> None:
    out = Path(out)
    ou = OU.isotropic(1.0, np.eye(2))
    jump = QuadraticRateJump(2, lam0=6.0, lam1=3.0, rho=0.5, jump=0.2)
    x0, T = np.array([0.3, 0.0]), 0.1
    log = ExperimentLog(out / "events_runs.jsonl")
    holder = {}

    def fac(b):
        f = P2BridgeFactory(ou, mode="truncated", race_kmax=3, max_factory_coins=5000, budget=b, certification="diagnostic")
        holder["f"] = f
        return f

    def fn(rng, budget, ctx):
        r = skeleton_run(ou, jump, fac, x0, T, rng, budget, ctx, refine_eps=0.3)
        r["p2_race_trials"] = [d.race_trials for d in holder["f"].diagnostics]
        r["p2_membership_coins"] = [d.membership_coins for d in holder["f"].diagnostics]
        r["tv_bound"] = max(holder["f"].tv_bounds) if holder["f"].tv_bounds else None
        return r

    rows = run_experiment("ou_chain_events_diagnostic", runs, fn, log, seed=77, max_seconds=300, verbose=True)
    s = summarize(rows)
    comp = [r["result"] for r in rows if r["status"] == "completed"]
    lines = [f"OU chain (diagnostic radii, truncated race kmax=3), T={T}, lambda=6+3|x|^2: requested {s['requested']}, completed {s['completed']}, "
             f"censored {s['censored']}, wall median {s.get('wall_q50', 0):.0f}s max {s.get('wall_max', 0):.0f}s.",
             "STATUS: approximate (truncated race, TV<= " + (f"{max((c['tv_bound'] or 0) for c in comp):.3f}" if comp else "n/a") + "), radii uncertified.", ""]
    for c in comp:
        lines.append(f"N_T={c['n_events']} taus={[round(t, 4) for t in c['taus']]} terminal={np.round(c['terminal'], 3).tolist()} "
                     f"refined radius={c.get('refined_radius')}  P2 trials/bridge={c['p2_race_trials']}")
    txt = "\n".join(lines)
    (out / "events_report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=24)
    main(ap.parse_args().runs)
