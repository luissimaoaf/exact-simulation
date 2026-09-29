# exactjd — exact jump skeletons, target-bridge correction and rough-path guided bridges

Research code implementing three connected papers (all by the same author):

* **Paper 1** – *Constructive ε-strong simulation of multidimensional guided diffusion bridges via rough paths*
* **Paper 2** – *ε-strong simulation of multidimensional diffusion bridges by retrospective acceptance–rejection*
* **Paper 3** – *Exact skeleton simulation of multivariate jump diffusions by localization and conditional reconstruction*

Python ≥ 3.11 (developed on 3.11), NumPy/SciPy/SymPy/mpmath. Install: `pip install -e .`; tests: `python -m pytest`.
(The repository root also contains unrelated legacy scripts `ajd/`, `jdcev/`, `sampling.py`, … — they are not part of `exactjd`.)

## The three output contracts (never conflated)

| Paper | What the code returns | What it is **not** |
|---|---|---|
| 1 | finite path + certified radius `e_n` for one **retained latent guided bridge** (`Q^{u,v}`) | not the target bridge law |
| 2 | same object, latent path selected with the **exact target-bridge law** by retrospective acceptance–rejection | not exact coordinates; Ψ-selection exact, path still ε-strong |
| 3 | **event times, marks, pre/post-jump states, terminal state (exact primitives)** + refinable segments | segments are only ε-strong; candidate/bridge interiors are never "exact coordinates" |

"Exact" always means *under the papers' ideal-arithmetic assumptions*. The floating-point implementation is documented
separately: see **Arithmetic caveats** below and `docs/CHECKLIST.md`.

## Layout

```
exactjd/
  rng.py                 independent streams (SeedSequence), exact LazyUniform, work Budget / BudgetExhausted
  models/                Diffusion / transition / bridge / jump interfaces; OU (multivariate, closed-form checks),
                         nonlinear 2-d benchmark (direct transformed reference), jump mechanisms
  baselines/euler.py     Euler(-jump) discretisations — APPROXIMATE, clearly labelled
  brownian/              tail-first retained Brownian record, m=2 Lévy-area enhanced driver, canonical lift, certificates
  bridges/               additive.py (Paper 1 §10 first-level route), cpverify.py / rough_guided.py / coeffbounds.py /
                         roughmodels.py (general rough route: block certificate, tube, terminal closure, interval bounds)
  correction/            abstract.py (Alg 3.6 race), factory.py, moment.py (M_c), bridge.py (OU target bridges),
                         nonlinear.py (guided→correction for the nonlinear benchmark, DIAGNOSTIC)
  jump/                  skeleton.py (Algorithms A–D + bounded baseline), records.py (bridge-record factories D3)
  experiments/           runner.py (resumable, censoring-aware) and the milestone experiment scripts
tests/                   unit + statistical validation (marked `slow` where >1 min)
docs/                    PLAN.md (dependency map), CHECKLIST.md (status), HESTON_CONTRACTS.md
results/                 produced tables/plots (`AB`, `CD`, `E`, `chain`, `heston`, `demo`)
```

## Reproducing each milestone

| Milestone | Code | Validation | Example / experiment |
|---|---|---|---|
| A models & baselines | `models/`, `baselines/` | `tests/test_models.py` | `python -m exactjd.experiments.exp_AB_reference` → `results/AB` |
| B Brownian + rough driver (BCD-type) | `brownian/` | `tests/test_brownian.py` (Chen, area law, refinement, certificate soundness) | same script (certificate scaling table/plot) |
| C guided bridges (Paper 1) | `bridges/` | `tests/test_additive_bridge.py`, `tests/test_rough.py` | `python -m exactjd.experiments.exp_CD_guided_race` → `results/CD` |
| D target-bridge correction (Paper 2) | `correction/` | `tests/test_correction.py` | same script (race pilots, censored runs) |
| E jump diffusion (Paper 3) | `jump/` | `tests/test_jump.py`, `tests/test_runner.py` | `python -m exactjd.experiments.exp_E_jump` → `results/E`; **headline demo** `python -m exactjd.experiments.demo_unbounded_skeleton` → `results/demo` |
| chains | `jump/records.py`, `correction/nonlinear.py` | `tests/test_nonlinear_chain.py` | `python -m exactjd.experiments.exp_chain` → `results/chain` |
| Heston (conditional) | `bridges/roughmodels.py` | `tests/test_rough.py` | `python -m exactjd.experiments.exp_heston`, `docs/HESTON_CONTRACTS.md` |

Every experiment script has a small default pilot size and a `--full` flag. Long runs are resumable
(`experiments/runner.py`): one JSONL row per *requested* run (completed / censored / interrupted), pickled checkpoints at
event boundaries reproduce the original realisation, and `summarize` reports completion counts and runtime quantiles over
**all** requested runs. Censored runs are never dropped and surviving outputs are never presented as an exact sample.

## Headline example

`demo_unbounded_skeleton.py`: 3-dimensional OU jump diffusion with **globally unbounded intensity** `λ(x)=1+|x|²`,
exact jump skeleton (times, marks, pre/post states, terminal state) and two nested refinements of the retained continuous
segments (certified uniform radius 0.095 → 0.049; see `results/demo/skeleton.txt`, `skeleton.png`).
The primitives are the direct Gaussian ones of Paper 3 Prop. 6.2 (exact OU transitions, time-changed Brownian-bridge records).

## What is (not) achieved — short version

* **Paper 3 (Algorithms A–D)**: implemented and statistically validated against analytic references (first-event survival
  formula, Poisson counts, exact moments, agreement with global thinning). This is the strongest part.
* **Paper 2**: the abstract retrospective machinery is implemented and validated (lognormal tilt example, bounded route).
  The paper's fixed-envelope race has infinite mean work; pilots show most runs must be *censored* even for T=0.1 OU
  bridges. Its linear factory is implemented by an **independent construction with a proof** (Huber's exact algorithm is not
  reproduced in the paper text).
* **Paper 1**: the additive-noise route (first level only, Paper 1 §10) is implemented with separately tracked driver /
  solver / terminal errors. The **general rough-path certificate** (BCD driver ball → controlled-Picard blocks → terminal
  closure) is implemented but its constants are astronomically conservative: `ρ_α`-certificate `E_n<1` needs `n≈90`, and the
  block test needs ℓ≲1e-8 with per-block inflation `1/(1−λ_w)`, so it is only attainable for horizons ~1e-5. Certified runs
  raise `CertificateFailure` / `BudgetExhausted` explicitly.
* **Chains** (Paper 1 → 2 → 3) run only in **approximate** modes (support-truncated race with a Chebyshev TV bound, and/or
  calibrated *uncertified* radii), each flagged in the output.
* **Heston**: only Paper 1's interior guide + local bounds; full skeleton is conditional/missing (`docs/HESTON_CONTRACTS.md`).

## Arithmetic caveats (where floating point weakens formal certification)

Tail-first thresholds/probabilities, all certificate constants, interval Riemann sums of the likelihood, moment certificates
and the RK4/affine center solves are evaluated in double precision, not with outward rounding; the numerical solver error is
*estimated* (step doubling / Gauss-Legendre comparison), not certified; the tight OU `M_c` uses quadrature; sympy rationals
inside interval bounds print as floats. Nothing here is a machine-checked certificate. Equality events of probability zero
(ties in strict comparisons) are treated as impossible.

## Algorithm-to-code checklist
See `docs/CHECKLIST.md` (implemented / validated / approximate / conditional / missing).
