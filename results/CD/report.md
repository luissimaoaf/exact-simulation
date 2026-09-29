## C1. Additive-noise guided bridge (Paper 1 §10 route): certified radius by source of error

| T | level | total radius | driver (rigorous) | solver (estimate) | terminal (rigorous) | time |
|---|---|---|---|---|---|---|
| 1.0 | 6 | 5.951 | 5.561 | 1.3e-11 | 0.391 | 0.58s |
| 1.0 | 10 | 1.788 | 1.679 | 7.9e-13 | 0.109 | 0.02s |
| 1.0 | 14 | 0.5112 | 0.4814 | 4.8e-14 | 0.0297 | 0.16s |
| 1.0 | 18 | 0.142 | 0.134 | 3.9e-15 | 0.00805 | 2.58s |
| 0.25 | 6 | 1.462 | 1.269 | 5.1e-13 | 0.193 | 0.02s |
| 0.25 | 10 | 0.4376 | 0.3834 | 3.3e-14 | 0.0542 | 0.02s |
| 0.25 | 14 | 0.1248 | 0.1099 | 2.7e-15 | 0.0149 | 0.07s |
| 0.25 | 18 | 0.03461 | 0.03059 | 1.1e-15 | 0.00402 | 1.19s |


## C2. Rough route, nonlinear benchmark: certified driver ball (delta_n, R_n)

| level n | delta_n = E_n | R_n |
|---|---|---|
| 4 | 7.35e+03 | 1.97e+04 |
| 8 | 4.89e+03 | 1.48e+04 |
| 12 | 3.24e+03 | 1.15e+04 |
| 20 | 1.43e+03 | 7.85e+03 |
| 40 | 181 | 5.36e+03 |


## C2b. Largest block length certifiable by the controlled-Picard test (alpha=0.34), nonlinear benchmark

| R | max ell | lambda_w |
|---|---|---|
| 1.0 | 1e-08 | 0.452 |
| 2.0 | 1e-09 | 0.409 |
| 5.0 | 1e-10 | 0.474 |
| 20.0 | 1e-11 | 0.992 |

Each block multiplies the incoming radius by 1/(1-lambda_w) (eq. (31)); with ell<=1e-5 blocks the horizon T=1 would need >=1e5 blocks: enclosure inflation makes the certified tube unattainable at practical T (see tests for the attainable T~5e-6 regime).


Certified pipeline at T=1, level 8: **CertificateFailure: single fine step (ell=0.00195) fails: invariance not attained (R=1.48e+04, delta=4.89e+03)**


## C3. Heston (Paper 1 §9.3): interval bounds of ||D^r V|| on the box z in [0.6 z0, 1.4 z0]

| z0 | ||DV|| | ||D^3V|| | ||D^5V|| |
|---|---|---|---|
| 0.4 | 1.07 | 13.9 | 2.11e+03 |
| 0.2 | 1.51 | 78.5 | 4.77e+04 |
| 0.1 | 2.13 | 444 | 1.08e+06 |
| 0.05 | 3.01 | 2.51e+03 | 2.44e+07 |

Growth of D^5 ~ z^{-9/2} makes the local verifier's constants explode near the boundary (Paper 1 §10).


## D1. Paper 2 race on the lognormal example (Prop 5.5, c=T=1), paper's fixed log envelope, limits: 200k factory coins / 30 s

requested 20, completed 4, censored 16 (completion fraction 0.20); wall quantiles over ALL requested runs: median 0.00s, q90 1.24s, max 5.88s.
Completed runs are NOT an unbiased exact sample (censoring conditions on small factory scales). Raw trials per completed run: median 16 (theory 2D/Z=113).


## D2. OU target-bridge pilot, T=0.1: M_c(tight)=1.42, M_c(Thm 8.4)=26.8; requested 8, completed 2, censored 6; wall median 60.1s, max 61.3s (limit 60 s)


## D2. OU target-bridge pilot, T=0.25: M_c(tight)=1.89, M_c(Thm 8.4)=1.65e+03; requested 8, completed 0, censored 8; wall median 48.7s, max 61.3s (limit 60 s)
