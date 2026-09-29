## Heston guided variance (Paper 1 Prop 9.5), numerical check (approximate SDE simulation)

| quantity | value | note |
|---|---|---|
| Y_tau at tau=14 (should -> 1) | 1.0000 ± 0.0000 | sd 0.0009 |
| min_tau Y_tau (positivity margin) | 0.0003 (min over 400 paths) | median 0.1627 |
| Feller ratio 2 kappa theta / xi^2 | 1.78 | >=1: interior regime |


## Interval bounds of ||D^r V|| for the Heston fields on boxes z in [0.6 z0, 1.4 z0]

| z0 | ||DV|| | ||D^3V|| | ||D^5V|| |
|---|---|---|---|
| 0.4 | 1.07 | 13.9 | 2.11e+03 |
| 0.2 | 1.51 | 78.5 | 4.77e+04 |
| 0.1 | 2.13 | 444 | 1.08e+06 |
| 0.05 | 3.01 | 2.51e+03 | 2.44e+07 |


The fifth-derivative bound grows ~ z0^{-9/2}: the local controlled-Picard verifier needs shells that are certified away from the boundary, and its constants degrade rapidly there (Paper 1 §10).

Missing contracts for a full Heston jump skeleton are listed in docs/HESTON_CONTRACTS.md.
