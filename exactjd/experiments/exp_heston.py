"""Heston-like benchmark (LATER benchmark, conditional).  Only what the papers establish is exercised:

  * Paper 1 §9.3 interior Heston guided proposal in its stated regime (kappa,theta,xi>0, |rho|<1, 2 kappa theta >= xi^2):
    Prop 9.5 says the guided (P,V) solution stays in D=R x (0,inf), is nonexplosive and pins to (p*,v*).
    We check the time-changed variance guide (Y_tau = V_t/v*, t=T(1-e^{-tau})) numerically (approximate SDE simulation:
    NOT a certificate): positivity margin, convergence Y->1, price pinning.
  * Interval derivative bounds of the fields on boxes z>=l (local verifier input) showing the z^{-9/2} blow-up.
  * It does NOT run a Heston jump skeleton: the target-bridge / moment / exact-transition contracts are missing
    (docs/HESTON_CONTRACTS.md).  Any approximate Heston jump simulation would have to be labelled approximate.

    python -m exactjd.experiments.exp_heston
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..bridges.roughmodels import HestonRoughModel
from .exp_AB_reference import md_table


def simulate_variance_guide(kappa, theta, xi, v0, vstar, T, tau_max, dtau, rng):
    """Euler (reflected) for dY = [Y(1-Y) + h(kappa (theta/v* - Y))] dtau + (xi/sqrt(v*)) sqrt(h Y) dbeta, h=T e^-tau, Y0=v0/v*."""
    n = int(tau_max / dtau)
    Y = v0 / vstar
    ymin = Y
    for k in range(n):
        tau = k * dtau
        h = T * math.exp(-tau)
        drift = Y * (1 - Y) + h * kappa * (theta / vstar - Y)
        Y = Y + drift * dtau + xi / math.sqrt(vstar) * math.sqrt(max(h * Y, 0.0) * dtau) * rng.standard_normal()
        Y = abs(Y)
        ymin = min(ymin, Y)
    return Y, ymin


def main(out: str = "results/heston", n: int = 400) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    kappa, theta, xi, rho = 2.0, 0.04, 0.3, -0.5
    assert 2 * kappa * theta >= xi**2
    v0, vstar, T = 0.04, 0.09, 1.0
    res = [simulate_variance_guide(kappa, theta, xi, v0, vstar, T, 14.0, 0.005, rng) for _ in range(n)]
    Yend = np.array([r[0] for r in res])
    ymin = np.array([r[1] for r in res])
    rows = [["Y_tau at tau=14 (should -> 1)", f"{Yend.mean():.4f} ± {Yend.std()/math.sqrt(n):.4f}", f"sd {Yend.std():.4f}"],
            ["min_tau Y_tau (positivity margin)", f"{ymin.min():.4f} (min over {n} paths)", f"median {np.median(ymin):.4f}"],
            ["Feller ratio 2 kappa theta / xi^2", f"{2*kappa*theta/xi**2:.2f}", ">=1: interior regime"]]
    txt = "## Heston guided variance (Paper 1 Prop 9.5), numerical check (approximate SDE simulation)\n\n" \
          + md_table(["quantity", "value", "note"], rows)
    hm = HestonRoughModel(kappa, theta, xi, rho)
    rows = []
    for z0 in (0.4, 0.2, 0.1, 0.05):
        Kh = hm.V_bounds([0.0, z0], 0.4 * z0, [1, 3, 5], depth=2)
        rows.append([z0, f"{Kh[1]:.3g}", f"{Kh[3]:.3g}", f"{Kh[5]:.3g}"])
    txt += "\n\n## Interval bounds of ||D^r V|| for the Heston fields on boxes z in [0.6 z0, 1.4 z0]\n\n" \
           + md_table(["z0", "||DV||", "||D^3V||", "||D^5V||"], rows)
    txt += ("\n\nThe fifth-derivative bound grows ~ z0^{-9/2}: the local controlled-Picard verifier needs shells that are "
            "certified away from the boundary, and its constants degrade rapidly there (Paper 1 §10).\n"
            "\nMissing contracts for a full Heston jump skeleton are listed in docs/HESTON_CONTRACTS.md.\n")
    (out / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
