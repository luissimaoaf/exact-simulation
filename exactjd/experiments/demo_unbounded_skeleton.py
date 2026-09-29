"""Headline deliverable: exact jump skeleton of a 3-dimensional OU jump diffusion with GLOBALLY UNBOUNDED intensity
lambda(x)=lam0+lam1|x|^2, together with refinable continuous segments.

    python -m exactjd.experiments.demo_unbounded_skeleton [--seed 7] [--eps 0.1]

Prints event times, marks, pre-/post-jump states (exact primitives), the terminal state, the localization diagnostics of every
call of Algorithm C (label, local rate bound, candidate count, reconstruction trials), and refines the retained
segments twice (eps and eps/2) to show nested refinement of the SAME latent realisation.  Writes results/demo/*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..jump import skeleton as sk
from ..jump.records import OUBridgeFactory
from ..models.jump import QuadraticRateJump
from ..models.ou import OU
from ..rng import Budget


def main(seed: int = 7, eps: float = 0.1, out: str = "results/demo") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d, T = 3, 1.0
    Sigma = np.array([[1.0, 0.3, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 0.9]])
    ou = OU.isotropic(1.0, Sigma)
    jump = QuadraticRateJump(d, lam0=1.0, lam1=1.0, rho=0.5, jump=0.2)
    x0 = np.array([1.0, 0.5, -0.5])
    budget = Budget()
    rng = np.random.default_rng(seed)
    skel = sk.simulate_skeleton(ou, jump, OUBridgeFactory(ou, budget), x0, T, rng, budget=budget)
    lines = [f"d={d}, T={T}, x0={x0.tolist()}, lambda(x)=1+|x|^2 (unbounded), update x -> 0.5 x + Z, |Z_j|=0.2",
             f"N_T = {skel.n_events} events;  primitive calls: {budget.calls}", ""]
    for k, e in enumerate(skel.events, 1):
        lines.append(f"event {k}: tau={e.tau:.6f}  pre={np.round(e.pre, 4).tolist()}  mark={np.round(e.mark, 2).tolist()}  post={np.round(e.post, 4).tolist()}")
    lines.append(f"terminal X_T = {np.round(skel.terminal, 4).tolist()}\n")
    lines.append("call | theta  label  ell(local rate bound)  candidates  recon.trials  bridges built")
    for i, sg in enumerate(skel.segments):
        f = sg.info
        lines.append(f"{i+1:>4} | {f.theta:.3f}  {f.label:>3}    {f.ell:9.3f}          {f.n_candidates:>5}       {f.skeleton_trials:>5}        {f.bridges_built:>5}")
    p1 = skel.refine(eps)
    p2 = skel.refine(eps / 2)
    tt, v1 = p1.grid(200)
    _, v2 = p2.grid(200)
    lines.append(f"\nrefine({eps}): certified uniform radius {p1.radius:.4f};  refine({eps/2}): radius {p2.radius:.4f};  "
                 f"max |path_eps - path_eps/2| = {np.max(np.linalg.norm(v1 - v2, axis=1)):.4f} <= {p1.radius + p2.radius:.4f}")
    txt = "\n".join(lines)
    print(txt)
    (out / "skeleton.txt").write_text(txt)
    json.dump(dict(seed=seed, x0=x0.tolist(), T=T, events=[dict(tau=e.tau, pre=e.pre.tolist(), mark=np.asarray(e.mark).tolist(),
                   post=e.post.tolist()) for e in skel.events], terminal=skel.terminal.tolist(), radius=p2.radius),
              open(out / "skeleton.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
        for j in range(d):
            axs[0].plot(tt, v2[:, j], label=f"$X^{{({j+1})}}$")
        for e in skel.events:
            axs[0].axvline(e.tau, color="k", lw=0.6, ls=":")
        axs[0].legend(ncol=3, fontsize=8)
        axs[0].set_title(f"Exact skeleton (N_T={skel.n_events}); eps-strong path, radius {p2.radius:.3f}", fontsize=10)
        lam = [jump.intensity(t, v) for t, v in zip(tt, v2)]
        axs[1].plot(tt, lam, "r")
        axs[1].set_ylabel("$\\lambda(X_t)$ (unbounded)"); axs[1].set_xlabel("t")
        fig.tight_layout(); fig.savefig(out / "skeleton.png", dpi=130)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eps", type=float, default=0.1)
    a = ap.parse_args()
    main(a.seed, a.eps)
