## A. Constant rate lambda=2, additive update (d=2, T=1, x0=(0.5,-0.5))

| sampler | N | E[N_T] ± se | E[X1_T] | Var X1_T | Var X2_T |
|---|---|---|---|---|---|
| localized (Alg C-D) | 1500 | 1.987 ± 0.036 | +0.193 | 0.427 | 0.462 |
| bounded thinning (§3) | 8000 | 2.025 ± 0.016 | +0.180 | 0.467 | 0.467 |
| Euler baseline (64 steps, approx.) | 1500 | 1.929 ± 0.036 | +0.180 | 0.473 | 0.466 |
| analytic | - | 2.000 | +0.184 | 0.467 | 0.467 |

Localized runs: requested 1500, completed 1500, censored 0; wall median 0.230s, q99 1.351s.


## B. Bounded state-dependent rate 1+3|x|²/(1+|x|²) (deterministic bound 4)

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| P(N_T=k) localized | 0.139 | 0.301 | 0.263 | 0.171 | 0.082 | 0.031 | 0.009 | 0.002 |
| P(N_T=k) global thinning | 0.133 | 0.298 | 0.285 | 0.172 | 0.075 | 0.028 | 0.007 | 0.002 |

N_localized=1500, N_global=8000; total variation distance 0.023 (sampling noise level ~0.037)


## C. Unbounded quadratic intensity lambda=1+lam1|x|² (globally unbounded), contractive update

First-event survival P(tau_1>t): empirical [95% Wilson] vs analytic (App. E.1).

| lam1 | N | t=0.25 | t=0.5 | t=1 | E[N_T] | mean recon. trials | max trials | label counts | mean candidates | wall q50/q99 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1500 | 0.739 [0.716,0.760] vs 0.750 ✓ | 0.548 [0.523,0.573] vs 0.558 ✓ | 0.302 [0.279,0.326] vs 0.305 ✓ | 1.15 | 3.91 | 3944 | [2141, 1069, 22] | 1.6 | 0.16/1.30 |
| 1.0 | 1500 | 0.661 [0.636,0.684] vs 0.672 ✓ | 0.439 [0.414,0.464] vs 0.441 ✓ | 0.188 [0.169,0.209] vs 0.187 ✓ | 1.54 | 5.23 | 5942 | [2522, 1250, 37] | 3.7 | 0.23/1.91 |
| 4.0 | 1500 | 0.457 [0.432,0.482] vs 0.449 ✓ | 0.208 [0.188,0.229] vs 0.205 ✓ | 0.054 [0.044,0.067] vs 0.045 ✓ | 2.74 | 2.66 | 2264 | [3963, 1617, 31] | 11.2 | 0.35/2.70 |


Euler jump-diffusion BASELINE, lam1=1 (approximate hazard): E[N_T] by grid size

| steps | E[N_T] |
|---|---|
| 32 | 1.523 ± 0.049 |
| 128 | 1.563 ± 0.047 |
| 512 | 1.553 ± 0.050 |
