"""Milestones A/B experiments: reference simulators, Brownian/OU checks, tail-first records, Levy-area certificates.

    python -m exactjd.experiments.exp_AB_reference [--full]

Outputs (results/AB/): ou_checks.md, euler_sensitivity.md, brownian_certificates.md, rough_certificate_scaling.md,
rough_certificate_scaling.png.  All statistics report sample sizes and standard errors.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from ..baselines.euler import euler_maruyama
from ..brownian import rough
from ..brownian.levyarea import BrownianAreaRecord
from ..brownian.record import BrownianRecord
from ..models.ou import OU


def md_table(header, rows) -> str:
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main(full: bool = False, out: str = "results/AB") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    N = 20000 if full else 4000

    # ---------------------------------------------------------------- OU exact vs analytic
    B = np.array([[-1.0, 0.4], [-0.2, -0.5]])
    ou = OU(B, [0.3, -0.1], np.array([[1.0, 0.2], [0.0, 0.8]]))
    x0, T = np.array([0.5, -1.0]), 1.0
    mu, Q = ou.mean_cov(T, x0)
    ys = np.array([ou.sample_transition(0, x0, T, rng) for _ in range(N)])
    se = np.sqrt(np.diag(Q) / N)
    rows = [["mean_1", f"{ys[:, 0].mean():.4f} ± {se[0]:.4f}", f"{mu[0]:.4f}"],
            ["mean_2", f"{ys[:, 1].mean():.4f} ± {se[1]:.4f}", f"{mu[1]:.4f}"],
            ["var_1", f"{ys[:, 0].var():.4f}", f"{Q[0, 0]:.4f}"], ["var_2", f"{ys[:, 1].var():.4f}", f"{Q[1, 1]:.4f}"],
            ["cov_12", f"{np.cov(ys.T)[0, 1]:.4f}", f"{Q[0, 1]:.4f}"]]
    (out / "ou_checks.md").write_text(f"Exact OU transition, N={N}\n\n" + md_table(["statistic", "sample ± se", "analytic"], rows))

    # ---------------------------------------------------------------- Euler baseline sensitivity (approximate!)
    rows = []
    Ne = 1500 if full else 400
    for k in (2, 4, 6, 8):
        steps = 2**k
        v = np.array([euler_maruyama(ou, x0, T, steps, rng)[1][-1] for _ in range(Ne)])
        rows.append([steps, f"{v[:, 0].var():.4f}", f"{Q[0, 0]:.4f}", f"{v[:, 0].mean() - mu[0]:+.4f}"])
    (out / "euler_sensitivity.md").write_text(
        f"BASELINE (Euler-Maruyama, approximate) N={Ne}: grid sensitivity of terminal variance/mean bias\n\n"
        + md_table(["steps", "var_1 (Euler)", "var_1 (exact)", "mean_1 bias"], rows))

    # ---------------------------------------------------------------- tail-first Brownian record certificates
    rows = []
    recs = [BrownianRecord(1.0, 2, np.random.default_rng(s)) for s in range(200)]
    Ks = np.array([r.Kstar for r in recs])
    for n in (4, 8, 12, 16, 20):
        rad = np.array([r.radius(n) for r in recs])
        fine = []
        for r in recs[:40]:
            nd, poly = r.nodes(14), r.polygon(min(n, 14))
            tt = np.linspace(0, 1, len(nd))
            fine.append(np.max(np.linalg.norm(nd - poly(tt), axis=1)) if n < 14 else 0.0)
        rows.append([n, f"{np.median(rad):.4f}", f"{rad.max():.4f}", f"{np.mean(fine):.4f}" if n < 14 else "-"])
    (out / "brownian_certificates.md").write_text(
        "Tail-first Brownian record: certified sup error radius(n) over 200 records (d=2, T=1); "
        f"K* distribution counts: {np.bincount(Ks).tolist()}; last column = mean observed distance to level-14 proxy (40 records)\n\n"
        + md_table(["level n", "median radius", "max radius", "observed (proxy)"], rows))

    # ---------------------------------------------------------------- rough-path certificate scaling
    ar = BrownianAreaRecord(1.0, np.random.default_rng(3))
    alpha, abar, beta = 0.34, 0.49, 0.6
    rows, ns, Es = [], [], []
    for n in (4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 120):
        c = ar.constants(alpha, abar, beta, n)
        ns.append(n)
        Es.append(c["E"])
        rows.append([n, f"{c['H1']:.3g}", f"{c['H2']:.3g}", f"{c['G']:.3g}", f"{c['E']:.3g}", f"{c['Nnorm']:.3g}"])
    need = {}
    for target in (1.0, 0.1, 0.01):
        nn = next((n for n in range(4, 400) if ar.constants(alpha, abar, beta, n)["E"] < target), None)
        need[target] = nn
    txt = (f"Certified rho_alpha(W, W^[n]) <= E_n (Theorem 4.2/Lemma 4.1 with alpha={alpha}, abar={abar}, beta={beta}), one record.\n\n"
           + md_table(["n", "H1", "H2", "G_n", "E_n", "N_n"], rows)
           + f"\nSmallest n with E_n < 1, 0.1, 0.01: {need}  (mesh 2^-n; storage 2^n points) -- the certificate is astronomically conservative.\n")
    # empirical residual identity check + area moments
    A, C = [], []
    for s in range(300 if not full else 1500):
        r = BrownianAreaRecord(1.0, np.random.default_rng(1000 + s))
        nd = r.nodes(8)
        A.append(rough.area(nd, 0, len(nd) - 1))
        C.append(rough.chen_defect(nd, 3, 50, 200))
    A = np.array(A)
    txt += (f"\nLevy area of the retained hierarchy (level 8), N={len(A)}: mean {A.mean():+.4f} ± {A.std()/math.sqrt(len(A)):.4f}, "
            f"var {A.var():.4f} (theory {0.25*(1-2**-8):.4f}), max Chen defect {max(C):.2e}\n")
    (out / "rough_certificate_scaling.md").write_text(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.semilogy(ns, Es, "o-")
        ax.axhline(1.0, color="gray", ls="--")
        ax.set_xlabel("dyadic level n"); ax.set_ylabel("certified rho_alpha bound E_n")
        ax.set_title("Paper 1 driver certificate (one retained path)")
        fig.tight_layout(); fig.savefig(out / "rough_certificate_scaling.png", dpi=130)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
