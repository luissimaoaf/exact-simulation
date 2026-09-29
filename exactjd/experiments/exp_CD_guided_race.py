"""Milestones C/D experiments: guided bridges (Paper 1) and target-bridge correction (Paper 2).

    python -m exactjd.experiments.exp_CD_guided_race [--full]

C: additive-noise route: certified radius (driver / solver / terminal) vs level; rough-path route: driver ball,
   block certificate feasibility (explicit failures), Heston coefficient blow-up.
D: Paper 2 race pilots with the paper's fixed log envelope (completion counts, work quantiles; censored runs are
   reported, never dropped), the support-truncated TEST HARNESS law check, OU target-bridge pilots.
Outputs in results/CD/.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ..bridges.additive import AdditiveGuidedBridge, AdditiveModel
from ..bridges.cpverify import CertificateFailure, Bounds, sewing_constant, verify_block
from ..bridges.rough_guided import RoughGuidedBridge
from ..bridges.roughmodels import HestonRoughModel, NonlinearRoughModel
from ..correction.abstract import LogEnvelope, retrospective_select
from ..correction import bridge as cb
from ..correction import moment as mom
from ..models.ou import OU
from ..rng import Budget, BudgetExhausted
from .exp_AB_reference import md_table
from .runner import ExperimentLog, run_experiment, summarize


class GaussTilt:
    """Paper 2 Prop 5.5 example: J=-cR-c^2T, R~N(0,T); target R~N(-cT,T); Z=exp(-c^2T/2)."""

    def __init__(self, c=1.0, T=1.0):
        self.c, self.T = c, T

    def new_proposal(self, rng):
        R = math.sqrt(self.T) * rng.standard_normal()

        class P:
            J = -self.c * R - self.c**2 * self.T
            r = R

            def interval(s, n):
                w = 3.0 * 2.0 ** (-n)
                return s.J - w, s.J + w
        return P()


def main(full: bool = False, out: str = "results/CD") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    txt = []

    # ------------------------------------------------------------------ C1 additive route
    ou = OU(np.array([[-1.0, 0.3], [-0.2, -0.5]]), [0.2, 0.0], np.eye(2))
    model = AdditiveModel.from_ou(ou)
    rows = []
    for T in (1.0, 0.25):
        br = AdditiveGuidedBridge(model, np.zeros(2), np.array([0.3, -0.2]) if T < 1 else np.array([1.0, -1.0]), T,
                                  np.random.default_rng(1))
        for n in (6, 10, 14, 18):
            t0 = time.time()
            e = br.enclosure(n)
            rows.append([T, n, f"{e.radius:.4g}", f"{e.parts['driver']:.4g}", f"{e.parts['solver']:.2g}",
                         f"{e.parts['terminal']:.3g}", f"{time.time()-t0:.2f}s"])
    txt.append("## C1. Additive-noise guided bridge (Paper 1 §10 route): certified radius by source of error\n\n"
               + md_table(["T", "level", "total radius", "driver (rigorous)", "solver (estimate)", "terminal (rigorous)", "time"], rows))

    # ------------------------------------------------------------------ C2 rough route (nonlinear benchmark)
    rows = []
    for n in (4, 8, 12, 20, 40):
        br = RoughGuidedBridge(NonlinearRoughModel(), [0.0, 0.0], [0.6, 0.2], 1.0, np.random.default_rng(3), budget=Budget())
        d = br.driver(n)
        rows.append([n, f"{d['delta']:.3g}", f"{d['R']:.3g}"])
    txt.append("\n## C2. Rough route, nonlinear benchmark: certified driver ball (delta_n, R_n)\n\n"
               + md_table(["level n", "delta_n = E_n", "R_n"], rows))
    # attainable regime of the block certificate (single block): largest ell passing for given (R, delta)
    model_r = NonlinearRoughModel().bind([0.6, 0.2], 1.0)
    K = model_r.V_bounds([0.0, 0.0], 3.0, range(0, 4))
    bounds = Bounds((K[0], K[1], K[2], K[3]), (0, 0, 0), 0.0, 0.0)
    rows = []
    ell_R1 = None
    for R in (1.0, 2.0, 5.0, 20.0):
        ok_ell = None
        for e in range(-1, -16, -1):
            ell = 10.0**e
            r = verify_block(lambda rad: bounds, ell, 0.34, R * ell**0.34, R**2 * ell ** 0.68, 0.0, 0.0)
            if r.ok:
                ok_ell = ell
                lam = r.lam
                break
        ell_R1 = ok_ell if (R == 1.0) else ell_R1
        rows.append([R, f"{ok_ell:.0e}" if ok_ell else "none >1e-15", f"{lam:.3f}" if ok_ell else "-"])
    txt.append("\n## C2b. Largest block length certifiable by the controlled-Picard test (alpha=0.34), nonlinear benchmark\n\n"
               + md_table(["R", "max ell", "lambda_w"], rows)
               + f"\nEach block multiplies the incoming radius by 1/(1-lambda_w) (eq. (31)); with blocks of length ell<={ell_R1 or float('nan'):.0e} (R=1) "
                 f"the horizon T=1 needs >={1.0/(ell_R1 or 1e-300):.0e} blocks: enclosure inflation makes the certified tube unattainable at practical T "
                 "(tests cover the attainable tiny-T regime).\n")
    try:
        br = RoughGuidedBridge(NonlinearRoughModel(), [0.0, 0.0], [0.6, 0.2], 1.0, np.random.default_rng(3), budget=Budget())
        br.certified_enclosure(8, max_blocks=300)
        outcome = "certified (unexpected)"
    except (BudgetExhausted, CertificateFailure) as e:
        outcome = f"{type(e).__name__}: {e}"
    txt.append(f"\nCertified pipeline at T=1, level 8: **{outcome}**\n")

    # ------------------------------------------------------------------ C3 Heston coefficient blow-up
    hm = HestonRoughModel(2.0, 0.04, 0.3, -0.5)
    rows = []
    for z in (0.4, 0.2, 0.1, 0.05):
        Kh = hm.V_bounds([0.0, z], 0.4 * z, [1, 3, 5], depth=2)
        rows.append([z, f"{Kh[1]:.3g}", f"{Kh[3]:.3g}", f"{Kh[5]:.3g}"])
    txt.append("\n## C3. Heston (Paper 1 §9.3): interval bounds of ||D^r V|| on the box z in [0.6 z0, 1.4 z0]\n\n"
               + md_table(["z0", "||DV||", "||D^3V||", "||D^5V||"], rows)
               + "\nGrowth of D^5 ~ z^{-9/2} makes the local verifier's constants explode near the boundary (Paper 1 §10).\n")

    # ------------------------------------------------------------------ D1 race pilots (paper's fixed envelope)
    log = ExperimentLog(out / "race_pilots.jsonl")
    n_runs = 60 if full else 20
    Mc = math.exp(-0.5) * 2.4
    env = LogEnvelope(Mc)

    def fn(rng, budget, ctx):
        sel = retrospective_select(GaussTilt(1.0, 1.0), env, rng, budget=budget, max_factory_coins=200_000)
        d = sel.diagnostics
        return dict(R=sel.proposal.r, trials=d.race_trials, mem_coins=d.membership_coins, factory_coins=d.factory_coins,
                    max_log_C=d.max_log_C, band=d.selected_band, finals=d.final_attempts)
    rows_ = run_experiment("race_lognormal_log_envelope", n_runs, fn, log, seed=11, max_seconds=30, verbose=False)
    s = summarize(rows_)
    comp = [r["result"] for r in rows_ if r["status"] == "completed"]
    txt.append("\n## D1. Paper 2 race on the lognormal example (Prop 5.5, c=T=1), paper's fixed log envelope, limits: 200k factory coins / 30 s\n\n"
               f"requested {s['requested']}, completed {s['completed']}, censored {s['censored']} (completion fraction {s['completion_fraction']:.2f}); "
               f"wall quantiles over ALL requested runs: median {s.get('wall_q50', 0):.2f}s, q90 {s.get('wall_q90', 0):.2f}s, max {s.get('wall_max', 0):.2f}s.\n"
               "Completed runs are NOT an unbiased exact sample (censoring conditions on small factory scales). "
               f"Raw trials per completed run: median {np.median([c['trials'] for c in comp]) if comp else float('nan'):.0f} "
               f"(theory 2D/Z={(16+36*Mc)/math.exp(-0.5):.0f}).\n")

    # ------------------------------------------------------------------ D2 OU target-bridge pilots
    log2 = ExperimentLog(out / "ou_bridge_pilots.jsonl")
    for T, v in ((0.1, np.array([0.1, 0.0])), (0.25, np.array([0.3, -0.2]))):
        orc = cb.OUGuidedOracle(ou, np.zeros(2), v, T)
        tight = orc.layer_certificate("tight")
        thm = orc.layer_certificate("thm84")

        def fn2(rng, budget, ctx, orc=orc, tight=tight):
            tb = cb.sample_target_bridge(orc, rng, Mc=tight["M_c"], budget=budget, max_factory_coins=20000)
            d = tb.selection.diagnostics
            e = tb.refine_to(0.5, n0=5)
            return dict(trials=d.race_trials, coins=d.membership_coins, band=d.selected_band, level=d.max_refine_level,
                        radius=e.radius)
        cfg = f"ou_bridge_T{T}"
        rows_ = run_experiment(cfg, 8 if not full else 30, fn2, log2, seed=5, max_seconds=60, verbose=False)
        s = summarize(rows_)
        txt.append(f"\n## D2. OU target-bridge pilot, T={T}: M_c(tight)={tight['M_c']:.3g}, M_c(Thm 8.4)={thm['M_c']:.3g}; "
                   f"requested {s['requested']}, completed {s['completed']}, censored {s['censored']}; "
                   f"wall median {s.get('wall_q50', 0):.1f}s, max {s.get('wall_max', 0):.1f}s (limit 60 s)\n")

    (out / "report.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
