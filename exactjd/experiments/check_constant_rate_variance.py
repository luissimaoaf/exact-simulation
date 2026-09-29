"""Large-sample exactness check: constant-rate benchmark terminal variance via the LOCALIZED sampler (Algorithms C-D).

    python -m exactjd.experiments.check_constant_rate_variance [--n 6000] [--workers 3]

Analytic reference (Paper 3 App. E.2 A): Var X_T^{(j)} = q (1 + lam jump^2), q=(1-e^{-2 kappa T})/(2 kappa), E N_T = lam T.
Independent SeedSequence streams per worker/run; result -> results/E/constant_rate_variance.md
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..jump import skeleton as sk
from ..jump.records import OUBridgeFactory
from ..models.jump import ConstantRateJump
from ..models.ou import OU
from ..rng import StreamFactory


def _chunk(args):
    lo, hi, seed = args
    ou = OU.isotropic(1.0, np.eye(2))
    jump = ConstantRateJump(2, 2.0, 0.2)
    fac = OUBridgeFactory(ou)
    sf = StreamFactory(seed)
    out = []
    for i in range(lo, hi):
        s = sk.simulate_skeleton(ou, jump, fac, np.array([0.5, -0.5]), 1.0, sf.generator("cv", i), keep_records=False)
        out.append((s.n_events, *s.terminal))
    return out


def main(n: int = 6000, workers: int = 3, seed: int = 2024) -> None:
    bounds = np.linspace(0, n, workers * 4 + 1).astype(int)
    tasks = [(int(bounds[i]), int(bounds[i + 1]), seed) for i in range(len(bounds) - 1)]
    with ProcessPoolExecutor(workers) as ex:
        res = [r for chunk in ex.map(_chunk, tasks) for r in chunk]
    a = np.array(res)
    q = (1 - math.exp(-2)) / 2
    ref_var = q * (1 + 2.0 * 0.04)
    txt = [f"Localized sampler, constant rate 2 (N={len(a)} independent runs)", ""]
    txt.append(f"E[N_T] = {a[:, 0].mean():.4f} ± {a[:, 0].std()/math.sqrt(len(a)):.4f} (analytic 2)")
    for j in (0, 1):
        v = a[:, 1 + j].var(ddof=1)
        se = v * math.sqrt(2.0 / (len(a) - 1))
        m = a[:, 1 + j].mean()
        txt.append(f"X{j+1}_T: mean {m:+.4f} ± {a[:, 1+j].std()/math.sqrt(len(a)):.4f} (analytic {math.exp(-1)*[0.5,-0.5][j]:+.4f}); "
                   f"var {v:.4f} ± {se:.4f} (analytic {ref_var:.4f}, z={(v-ref_var)/se:+.2f})")
    out = "\n".join(txt)
    Path("results/E").mkdir(parents=True, exist_ok=True)
    Path("results/E/constant_rate_variance.md").write_text(out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    main(a.n, a.workers)
