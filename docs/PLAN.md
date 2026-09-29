# Dependency map and milestone plan

Papers: **P1** = constructive ε-strong guided bridges via rough paths; **P2** = ε-strong target bridges by
retrospective acceptance–rejection; **P3** = exact jump skeletons by localization + conditional reconstruction.

Three output contracts (never mixed):

| Paper | Output | Status of the output |
|---|---|---|
| P1 | finite path `X̂_n` + `e_n`, `‖X°−X̂_n‖∞ ≤ e_n`, latent `X° ~ Q^{u,v}` (guided law) | ε-strong enclosure, not target law |
| P2 | same object, latent `X⋆ ~ P^{u,v}` (target bridge) chosen by exact retrospective selection | ε-strong enclosure of an exactly-selected latent path; **no exact coordinates** |
| P3 | event times, marks, pre/post states, terminal state (exact primitives) + refinable segments | exact skeleton; segments only ε-strong |

## Algorithm table

| ID | Paper/§ | Inputs | Outputs | Depends on | Validation |
|---|---|---|---|---|---|
| B1 tail-first Brownian record | P1 §4 (BCD) | h, endpoints, m | nested dyadic values, certified `e_n`, Hölder constants | rng | Gaussian moments, refinement consistency, bound-vs-fine-path |
| B2 Lévy-area record (m=2) | P1 Thm 4.2, (18) | B1 | canonical polygonal lift, aggregate residual const `G`, `ρ_α` bound `E_r` | B1 | Chen, symmetry, area covariance, refinement consistency |
| C1 additive-noise guided bridge | P1 §10 | b Lipschitz, Σ const | Volterra path + certified `e_n` (first level only) | B1 | OU bridge exact reference, containment tests |
| C2 rough guided bridge | P1 §5–7 | b,σ (C⁵), B2 | CPVerify tube, TailClose, `E_pre + 2Mh^α` | B2 | soundness vs fine numerics; certificate scaling report |
| C3 localization | P1 §9 | open-domain data | shell search (Alg 2) | C2 | Heston Prop 9.5 containment sims |
| D1 lazy uniform, Huber factory, race, bands | P2 §3 | oracle, `M_c` | selected latent + record | rng | tilted-Gaussian law, lognormal example |
| D2 bounded-weight rejection | P2 Rem 3.9/§10 | valid bound | same | rng | same |
| D3 likelihood oracle | P2 §6 | C1/C2 | `J ∈ [J⁻,J⁺]` | C1/C2 | interval containment vs quadrature |
| D4 moment certificate | P2 Thm 8.4 | heat-kernel constants | `M_c` | model | Monte-Carlo sanity (not used for bound) |
| D5 target-bridge sampler | P2 Alg 7.1 | D1+D3+D4 | accepted record, refinable | all above | vs exact OU bridge |
| E0 bounded thinning | P3 §3 | L, exact transitions | skeleton | models | analytic moments, survival |
| E1 A: retrospective decision | P3 Alg A | record, L | 1{LU<Q} | C/D records | bias-free vs oracle |
| E2 B: label certification | P3 Alg B, Lem 4.1 | record, θ | layer I | bridge record | layer frequencies vs Monte Carlo |
| E3 C: next event | P3 Alg C | (s,x) | event or survival | E1,E2, D2, D3 | first-event survival formula (App. E.1) |
| E4 D: full skeleton | P3 Alg D | x0,T | skeleton + segment records | E3 | moments, counts, resume, refinement |

## Milestones (each: code + tests + runnable example)
A models/baselines → B Brownian/rough → C guided → D correction → E jump → F benchmarks, censored-run infrastructure, docs.
See `docs/CHECKLIST.md` for live status.
