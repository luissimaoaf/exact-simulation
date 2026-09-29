## OU chain (diagnostic radii, truncated race kmax=3), T=0.1, quadratic intensity

requested 6, completed 2, censored 4, wall median 16s / max 33s.  Completed skeletons: events [0, 0], refined radius [0.108, 0.108].
Censoring reasons: ['refinement level 20 exceeds MAX_LEVEL=19']

## OU chain (certified radii, truncated race kmax=3), T=0.1, quadratic intensity

requested 6, completed 0, censored 6, wall median 54s / max 170s.  Completed skeletons: events [], refined radius [].
Censoring reasons: ['refinement level 20 exceeds MAX_LEVEL=19']


## Nonlinear benchmark, DIRECT transformed route (not Papers 1-2): first-event survival, N=800

| t | empirical | analytic (Brownian coords, kappa=0) | z-score |
|---|---|---|---|
| 0.25 | 0.694 | 0.710 | -1.01 |
| 0.5 | 0.439 | 0.455 | -0.92 |
| 1.0 | 0.155 | 0.156 | -0.05 |


## Nonlinear benchmark: guided (Paper 1, rough route) -> correction (Paper 2), DIAGNOSTIC (uncertified J, truncated race)

requested 60, completed 39, censored 21; wall median 3.0s.

| law of X_{T/2} | N | mean ± se (x1, x2) | sd (x1, x2) |
|---|---|---|---|
| direct transformed reference | 2000 | +0.166 ± 0.012, +0.060 ± 0.008 | 0.523, 0.350 |
| guided proposal (uncorrected) | 60 | +0.192 ± 0.068, +0.003 ± 0.049 | 0.526, 0.377 |
| guided -> Paper 2 correction (diagnostic) | 39 | +0.090 ± 0.093, +0.058 ± 0.065 | 0.580, 0.404 |
