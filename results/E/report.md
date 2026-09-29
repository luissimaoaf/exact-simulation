## A. Constant rate lambda=2, additive update (d=2, T=1, x0=(0.5,-0.5))

| sampler | N | E[N_T] ± se | E[X1_T] | Var X1_T | Var X2_T |
|---|---|---|---|---|---|
| localized (Alg C-D) | 250 | 1.936 ± 0.085 | +0.216 | 0.421 | 0.365 |
| bounded thinning (§3) | 1500 | 2.021 ± 0.037 | +0.162 | 0.461 | 0.458 |
| Euler baseline (64 steps, approx.) | 250 | 2.084 ± 0.088 | +0.186 | 0.495 | 0.421 |
| analytic | - | 2.000 | +0.184 | 0.467 | 0.467 |

Localized runs: requested 250, completed 250, censored 0; wall median 0.230s, q99 1.558s.


## B. Bounded state-dependent rate 1+3|x|²/(1+|x|²) (deterministic bound 4)

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| P(N_T=k) localized | 0.116 | 0.304 | 0.288 | 0.176 | 0.084 | 0.028 | 0.004 | 0.000 |
| P(N_T=k) global thinning | 0.129 | 0.307 | 0.286 | 0.168 | 0.074 | 0.023 | 0.009 | 0.004 |

N_localized=250, N_global=1500; total variation distance 0.025 (sampling noise level ~0.089)


## C. Unbounded quadratic intensity lambda=1+lam1|x|² (globally unbounded), contractive update

First-event survival P(tau_1>t): empirical [95% Wilson] vs analytic (App. E.1).

| lam1 | N | t=0.25 | t=0.5 | t=1 | E[N_T] | mean recon. trials | max trials | label counts | mean candidates | wall q50/q99 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 250 | 0.744 [0.686,0.794] vs 0.750 ✓ | 0.564 [0.502,0.624] vs 0.558 ✓ | 0.320 [0.265,0.380] vs 0.305 ✓ | 1.10 | 1.80 | 33 | [350, 172, 3] | 1.6 | 0.16/0.76 |
| 1.0 | 250 | 0.648 [0.587,0.705] vs 0.672 ✓ | 0.388 [0.330,0.450] vs 0.441 ✓ | 0.148 [0.109,0.197] vs 0.187 ✓ | 1.62 | 2.11 | 102 | [437, 213, 4] | 3.7 | 0.20/1.80 |
| 4.0 | 250 | 0.500 [0.438,0.562] vs 0.449 ✓ | 0.248 [0.199,0.305] vs 0.205 ✓ | 0.052 [0.031,0.087] vs 0.045 ✓ | 2.72 | 1.95 | 103 | [655, 269, 5] | 11.2 | 0.33/1.65 |


Euler jump-diffusion BASELINE, lam1=1 (approximate hazard): E[N_T] by grid size

| steps | E[N_T] |
|---|---|
| 32 | 1.560 ± 0.094 |
| 128 | 1.647 ± 0.091 |
| 512 | 1.480 ± 0.091 |
