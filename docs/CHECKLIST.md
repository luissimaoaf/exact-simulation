# Algorithm-to-code checklist

Legend — **Impl**: code exists and runs · **Valid**: a validation was *executed* (test/experiment named) ·
**Approx**: implemented but not exact/certified (reason given) · **Cond**: conditional on inputs the papers do not establish ·
**Missing**: not implemented.  "Exact/certified" is always *under the papers' ideal-arithmetic assumptions*; floating-point
caveats are in README §"Arithmetic caveats".

## Milestone A — models, references
| Item | Status | Code | Validation |
|---|---|---|---|
| Interfaces (drift/σ, transition, bridge record, jump mechanism, local bounds) | Impl, Valid | `models/base.py` | used throughout |
| Brownian / multivariate OU exact transitions, Gaussian bridge values, moments | Impl, Valid | `models/ou.py` | `test_models.py` (closed forms, Euler agreement, bridge midpoint law) |
| First-event survival formula (Paper 3 App. E.1) | Impl, Valid | `OU.survival_quadratic_rate` | PDE residual test; used as reference in `test_jump.py`, `exp_E_jump` |
| Nonlinear 2-d benchmark: coefficients, exact transformed transitions, direct bridge reference | Impl, Valid | `models/nonlinear.py` | `test_models.py` (SDE law = Φ(BM), density normalisation) |
| Euler / Euler-jump baselines | Approx (baseline, labelled) | `baselines/euler.py` | grid sensitivity in `results/AB`, `results/E` |

## Milestone B — retained Brownian and rough driver
| Item | Status | Code | Validation |
|---|---|---|---|
| Nested dyadic Brownian/bridge record, refinement never changes revealed values | Impl, Valid | `brownian/record.py` | `test_brownian.py` (marginals, nesting, lazy-entry consistency) |
| Tail-first certificates (exact `K*` sampling, deterministic bounds on unrevealed levels): sup-error radius, increments, Hölder constant | Impl, Valid (ideal arithmetic; floats) | `brownian/tailfirst.py`, `record.py` | law of `K*`, exceed-conditioned vectors, radius/Hölder dominate fine proxy |
| Lévy-area second level (m=2) with tree-Gaussian aggregate-residual certificate; canonical polygonal lift, Chen's identity | Impl, Valid | `brownian/levyarea.py`, `rough.py` | Chen defect 1e-16; area law (var 1/4, kurtosis 5.15 vs 5); residual identity (eq. (18)); `G`, `H1,H2`, `E_n` dominate empirical values |
| `ρ_α` bound (Lemma 4.1 / Thm 4.2) | Impl, Valid; **constants extremely conservative** | `BrownianAreaRecord.constants` | `E_n<1` only for `n≈91` (results/AB) |
| BCD's own random-constant algorithm | **Not reproduced**: independent tail-first construction (own derivation, documented in module) | — | — |
| Driver dimension m>2 (joint area certificates) | **Missing** | — | needs joint conditioning over pairs |

## Milestone C — guided bridges (Paper 1)
| Item | Status | Code | Validation |
|---|---|---|---|
| Covariance-matched guide, Itô⇄Stratonovich correction `F=b−½ΣDσ_jσ_j` | Impl, Valid | `bridges/roughmodels.py` | `test_rough.py` (F=(q/2,0), Itô drift recovered, guided-law vs Euler guide) |
| Additive-noise route (Paper 1 §10, first level only): Volterra guide, Gronwall driver radius, terminal pinning `2Mh^q`, separate error parts | Impl, Valid (ideal arithmetic); solver error **estimated** | `bridges/additive.py` | `test_additive_bridge.py` (exact guided Gaussian moments, radii ⊇ proxy, RK4 vs affine solver) |
| Controlled-Picard block certificate (C.1–C.13), interval derivative bounds up to D⁵ | Impl, Valid **in its attainable regime** (T≈5e-6) | `bridges/cpverify.py`, `coeffbounds.py` | `test_rough.py` (radius ≥ true two-driver difference; bounds ≥ sampled derivatives) |
| Preterminal tube composition, terminal closure (Prop 6.4 form), adaptive refinement at the certified level | Impl; **certificate unattainable at practical T** (`CertificateFailure`/`BudgetExhausted`); tail closure not validated in a success regime | `bridges/rough_guided.py` | failure documented in `results/CD/report.md`, `test_certified_pipeline_reports_infeasibility_honestly` |
| Diagnostic enclosure for the rough route | Approx, flagged `diagnostic-UNCERTIFIED` | `RoughGuidedBridge.diagnostic_enclosure` | flagged test |
| Open-domain localization (Alg 2): box exhaustion, retrospective shell search, driver retained | Impl (shell search + local bounds); certified verifier inside a shell inherits the infeasibility | `bridges/localization.py` | `test_rough.py::test_open_domain_shell_search_heston…` |
| Interior Heston guide, Prop 9.5 containment | Impl; **numerical check only** | `roughmodels.HestonRoughModel`, `exp_heston.py` | Y→1, positivity (results/heston) |
| Full Heston target-bridge / moment / transition contracts | **Cond / Missing** | `docs/HESTON_CONTRACTS.md` | — |

## Milestone D — target-bridge correction (Paper 2)
| Item | Status | Code | Validation |
|---|---|---|---|
| Shifted-band membership coin (Alg 3.3), lazy uniforms, strict retrospective comparisons | Impl, Valid | `correction/abstract.py`, `rng.py` | law tests below |
| Linear Bernoulli factory (C>1) | Impl, Valid; **not Huber's algorithm** (own construction with proof, cost O(C/η²)+heavy tail) | `correction/factory.py` | bias, head property, `d≤π_n` inequality, Lemma 4.1 cost bound (`test_correction.py`) |
| Countable race, retained first-head proposal, final rejection (Alg 3.6) | Impl, Valid (support-truncated harness for the law; fixed envelope for censoring behaviour) | `abstract.py` | tilted-Gaussian law, mean trial count vs `2D/Z`; paper envelope: censored runs raise `BudgetExhausted` |
| Fixed log envelope race (paper) | Impl; **infinite mean work** — pilots mostly censored | `LogEnvelope` | `results/CD/report.md` (D1: 4/20 completed) |
| Power envelope (Prop 4.9), bounded-weight rejection route | Impl, Valid (bounded route law) | `PowerEnvelope`, `bounded_weight_select` | `test_correction.py` |
| Moment certificate Thm 8.4 (heat-kernel constants) | Impl (float); astronomically large for T≥0.25 | `correction/moment.py` | ordering tests; values in `results/CD` |
| Tight model-specific OU certificate `Z(1+√E_P J²)²` | Impl, Valid vs Monte Carlo; quadrature not interval-certified | `moment.ou_tight_certificate` | `test_correction.py` |
| Likelihood oracle from certified enclosures (interval Riemann + terminal tail), normaliser identity `E_Q Ψ=Z` | Impl, Valid (ideal arithmetic) | `correction/bridge.py` | `test_additive_bridge.py` |
| Accepted target bridge for OU with paper's race | Impl; **practically unusable** (level-20 / factory-scale censoring) | `sample_target_bridge` | pilots |
| Truncated-race target bridge | Approx: TV ≤ Chebyshev bound reported | `jump/records.py::P2BridgeFactory(mode='truncated')` | smoke test, chain runs |

## Milestone E — jump diffusion (Paper 3)
| Item | Status | Code | Validation |
|---|---|---|---|
| Bounded-intensity thinning (§3) | Impl, Valid | `jump/skeleton.py::bounded_skeleton` | Poisson counts, exact moments |
| Algorithm A retrospective decision | Impl, Valid | `retrospective_decision` | unbiasedness test |
| Algorithm B label certification (randomised layers, adaptive certified sup) | Impl, Valid | `layer_of`, `certify_reconstruction`, `records._AdaptiveSup` | sup intervals contain fine reference & nested |
| Conditional reconstruction with fixed shift/label/schedule (Lemma 4.2), skeleton filter | Impl, Valid | `reconstruct` | law at candidate times and T (`test_reconstruction_preserves_unconditional_law`) |
| Algorithm C next event; Algorithm D full skeleton with fresh restarts; zero intensity, T=0, tiny/long segments, d=1, random start | Impl, Valid | `next_event`, `simulate_skeleton` | analytic first-event survival (unbounded quadratic), constant-rate analytics, localized vs global thinning, edge tests |
| Refinable continuous segments, inactive continuation metadata, repeated refinement | Impl, Valid | `JumpSkeleton.refine` | nested/consistent refinement test |
| Checkpoint/resume, censored-run accounting | Impl, Valid | `experiments/runner.py` | identity of resumed realisation; interruption recorded |
| OU quadratic-intensity headline example (d=3) | Impl, Valid | `demo_unbounded_skeleton.py` | `results/demo` |
| Nonlinear benchmark, direct transformed route | Impl, Valid | `records.TransformedBridgeRecord` | survival vs analytic (`test_nonlinear_chain.py`) |
| Chain Paper 1→2→3 on OU | Approx (truncated race; diagnostic or certified-but-mostly-censored) | `P2BridgeFactory` | `results/chain` |
| Nonlinear guided→correction chain | Approx (diagnostic J-interval + truncated race) | `correction/nonlinear.py` | `results/chain` (law vs direct reference) |
| Exact-coordinate insertion into bridge records (D3+) | **Missing** (not needed; paper marks it optional) | — | — |

## Known gaps / obstacles (smallest missing primitives)
1. A **practical certified rough-path solver bound**: the paper's constants make `E_n<1` require `n≈90` and the block
   recursion inflate by `1/(1−λ_w)` per block; a sharper Lemma 4.1 / sewing analysis (or a Gronwall-type tube not built from
   short blocks) is the missing primitive for the multiplicative-noise chain.
2. **A finite-mean bridge-correction selection** (the paper proves the fixed-envelope race has infinite mean).
3. Certified **outward-rounded** arithmetic throughout (out of scope by request).
4. m>2 aggregate Lévy-area certificate; Heston target contracts (see `HESTON_CONTRACTS.md`).
