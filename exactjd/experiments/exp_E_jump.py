"""Milestone E benchmarks: OU jump diffusions with constant / bounded state-dependent / unbounded quadratic intensity.

    python -m exactjd.experiments.exp_E_jump [--full]

Exact samplers (bounded thinning, localized reconstruction, Algorithms A-D) are compared with
  * analytic references (Poisson counts, terminal moments, first-event survival formula, Paper 3 App. E),
  * BASELINE Euler jump-diffusion (approximate; grid sensitivity reported, never labelled exact).
Every requested run is logged (results/E/runs.jsonl); censored runs are counted, never dropped.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

from ..baselines.euler import euler_jump_diffusion
from ..jump import skeleton as sk
from ..jump.records import OUBridgeFactory
from ..models.jump import BoundedStateRateJump, ConstantRateJump, QuadraticRateJump
from ..models.ou import OU
from ..rng import Budget
from .exp_AB_reference import md_table
from .runner import ExperimentLog, run_experiment, skeleton_run, summarize


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(full: bool = False, out: str = "results/E") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = ExperimentLog(out / "runs.jsonl")
    n_loc = 1500 if full else 250
    n_glob = 8000 if full else 1500
    d, T = 2, 1.0
    ou = OU.isotropic(1.0, np.eye(d))
    x0 = np.array([0.5, -0.5])
    fac = lambda b: OUBridgeFactory(ou, b)
    rep = []

    def batch(cfg, jump, n, seed):
        def fn(rng, budget, ctx):
            return skeleton_run(ou, jump, fac, x0, T, rng, budget, ctx)
        rows = run_experiment(cfg, n, fn, log, seed=seed, max_seconds=120, verbose=False)
        return rows

    # ------------------------------------------------------------------ A: constant rate
    jA = ConstantRateJump(d, lam=2.0, jump=0.2)
    rowsA = batch("A_const_localized", jA, n_loc, 1)
    resA = [r["result"] for r in rowsA if r["status"] == "completed"]
    rng = np.random.default_rng(2)
    glob = [sk.bounded_skeleton(ou, jA, x0, T, rng) for _ in range(n_glob)]
    eul = [euler_jump_diffusion(ou, jA, x0, T, 64, rng) for _ in range(n_loc)]
    q = (1 - math.exp(-2)) / 2
    var_ref = q * (1 + 2.0 * 0.04)
    def stat(nev, xt):
        nev, xt = np.array(nev), np.array(xt)
        return [f"{nev.mean():.3f} ± {nev.std()/math.sqrt(len(nev)):.3f}", f"{xt[:, 0].mean():+.3f}",
                f"{xt[:, 0].var():.3f}", f"{xt[:, 1].var():.3f}"]
    rowsT = [["localized (Alg C-D)", len(resA)] + stat([r["n_events"] for r in resA], [r["terminal"] for r in resA]),
             ["bounded thinning (§3)", n_glob] + stat([s.n_events for s in glob], [s.terminal for s in glob]),
             ["Euler baseline (64 steps, approx.)", len(eul)] + stat([len(e[0]) for e in eul], [e[1] for e in eul]),
             ["analytic", "-", "2.000", f"{math.exp(-T)*x0[0]:+.3f}", f"{var_ref:.3f}", f"{var_ref:.3f}"]]
    rep.append("## A. Constant rate lambda=2, additive update (d=2, T=1, x0=(0.5,-0.5))\n\n"
               + md_table(["sampler", "N", "E[N_T] ± se", "E[X1_T]", "Var X1_T", "Var X2_T"], rowsA_fix(rowsT)))
    sA = summarize(rowsA)
    rep.append(f"Localized runs: requested {sA['requested']}, completed {sA['completed']}, censored {sA['censored']}; "
               f"wall median {sA['wall_q50']:.3f}s, q99 {sA['wall_q99']:.3f}s.\n")

    # ------------------------------------------------------------------ B: bounded state-dependent
    jB = BoundedStateRateJump(d, rho=0.5)
    rowsB = batch("B_bounded_localized", jB, n_loc, 3)
    resB = [r["result"] for r in rowsB if r["status"] == "completed"]
    rng = np.random.default_rng(4)
    globB = [sk.bounded_skeleton(ou, jB, x0, T, rng).n_events for _ in range(n_glob)]
    ks = np.arange(0, 8)
    pl = np.array([(np.array([r["n_events"] for r in resB]) == k).mean() for k in ks])
    pg = np.array([(np.array(globB) == k).mean() for k in ks])
    rep.append("\n## B. Bounded state-dependent rate 1+3|x|²/(1+|x|²) (deterministic bound 4)\n\n"
               + md_table(["k"] + [str(k) for k in ks], [["P(N_T=k) localized"] + [f"{p:.3f}" for p in pl],
                                                          ["P(N_T=k) global thinning"] + [f"{p:.3f}" for p in pg]])
               + f"\nN_localized={len(resB)}, N_global={n_glob}; total variation distance {0.5*np.abs(pl-pg).sum():.3f} "
                 f"(sampling noise level ~{0.5*math.sqrt(len(ks)/len(resB)):.3f})\n")

    # ------------------------------------------------------------------ C: unbounded quadratic rate
    rowsC_all = []
    for lam1 in (0.25, 1.0, 4.0):
        jC = QuadraticRateJump(d, lam0=1.0, lam1=lam1, rho=0.5)
        cfg = f"C_quadratic_lam1_{lam1}"
        rows = batch(cfg, jC, n_loc, 10 + int(lam1 * 4))
        res = [r["result"] for r in rows if r["status"] == "completed"]
        tau1 = np.array([r["taus"][0] if r["taus"] else math.inf for r in res])
        sv = []
        for t in (0.25, 0.5, 1.0):
            emp = (tau1 > t).mean()
            lo, hi = _wilson(int((tau1 > t).sum()), len(tau1))
            ref = ou.survival_quadratic_rate(t, x0, 1.0, lam1)
            sv.append(f"{emp:.3f} [{lo:.3f},{hi:.3f}] vs {ref:.3f}" + (" ✓" if lo <= ref <= hi else " ✗"))
        segs = [s for r in res for s in r["segments"]]
        s = summarize(rows)
        rowsC_all.append([lam1, len(res)] + sv + [f"{np.mean([r['n_events'] for r in res]):.2f}",
                          f"{np.mean([g['trials'] for g in segs]):.2f}", max(g["trials"] for g in segs),
                          f"{np.bincount([g['label'] for g in segs]).tolist()}",
                          f"{np.mean([g['cands'] for g in segs]):.1f}", f"{s['wall_q50']:.2f}/{s['wall_q99']:.2f}"])
    rep.append("\n## C. Unbounded quadratic intensity lambda=1+lam1|x|² (globally unbounded), contractive update\n\n"
               "First-event survival P(tau_1>t): empirical [95% Wilson] vs analytic (App. E.1).\n\n"
               + md_table(["lam1", "N", "t=0.25", "t=0.5", "t=1", "E[N_T]", "mean recon. trials", "max trials", "label counts",
                           "mean candidates", "wall q50/q99 (s)"], rowsC_all))

    # Euler baseline grid sensitivity for C (lam1=1): event-count mean
    jC = QuadraticRateJump(d, 1.0, 1.0, 0.5)
    rows = []
    for k in (5, 7, 9):
        rng = np.random.default_rng(20 + k)
        ev = [euler_jump_diffusion(ou, jC, x0, T, 2**k, rng) for _ in range(150 if not full else 600)]
        rows.append([2**k, f"{np.mean([len(e[0]) for e in ev]):.3f} ± {np.std([len(e[0]) for e in ev])/math.sqrt(len(ev)):.3f}"])
    rep.append("\nEuler jump-diffusion BASELINE, lam1=1 (approximate hazard): E[N_T] by grid size\n\n"
               + md_table(["steps", "E[N_T]"], rows))
    (out / "report.md").write_text("\n".join(rep))
    print("\n".join(rep))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        tt = np.linspace(0.02, 1.0, 40)
        ax.plot(tt, [ou.survival_quadratic_rate(t, x0, 1.0, 1.0) for t in tt], "k-", label="analytic (App. E.1)")
        rowsC = [r for r in log.rows() if r["config"] == "C_quadratic_lam1_1.0" and r["status"] == "completed"]
        tau1 = np.array([r["result"]["taus"][0] if r["result"]["taus"] else math.inf for r in rowsC])
        ax.step(tt, [(tau1 > t).mean() for t in tt], "r", where="post", label=f"exact skeleton (N={len(tau1)})")
        ax.set_xlabel("t"); ax.set_ylabel("P(first event > t)"); ax.legend(); ax.set_title("Unbounded quadratic intensity")
        fig.tight_layout(); fig.savefig(out / "survival_quadratic.png", dpi=130)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


def rowsA_fix(rows):
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
