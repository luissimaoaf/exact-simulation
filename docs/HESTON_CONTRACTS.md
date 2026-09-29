# Heston-like jump diffusion: what is established and what is missing

**Status: conditional.** Nothing here is presented as solved. Papers 1–3 do not establish the full contract stack for
Heston, and no Heston *jump skeleton* is produced by this package. What exists:

| Piece | Status | Where |
|---|---|---|
| Interior Heston guided proposal, non-strict Feller regime (`2κθ ≥ ξ²`, positive start/end variance) | proved in Paper 1 Prop 9.5 (compact containment in `R×(0,∞)`); **checked numerically** (approximate SDE simulation, not a certificate) | `experiments/exp_heston.py` |
| Guide coefficients (`σ, a, F = b − ½ΣDσ_jσ_j`, `A = a a_T^{-1} = (z/v⋆) I`) and Itô/Stratonovich conversion | implemented, unit tested | `bridges/roughmodels.py::HestonRoughModel`, `tests/test_rough.py` |
| Local interval derivative bounds up to `D^5` on boxes `z ≥ ℓ > 0` | implemented (interval arithmetic); shows the `z^{-9/2}` blow-up of `‖D^5 V‖` | `bridges/coeffbounds.py` |
| Open-domain shell selection (Paper 1 Alg 2) | logic follows `RoughGuidedBridge` with box bounds; certified tube/tail remain unattainable (same obstruction as the nonlinear benchmark) | — |

## Missing contracts for exact skeleton simulation (Paper 3) — to be identified/investigated, not assumed

1. **(D2) Exact transitions.** The CIR variance has an exact non-central χ² transition, but the joint `(log price, variance)`
   transition needs the integrated variance given `(V_s, V_t)`. Exact-in-ideal-arithmetic samplers exist (Broadie–Kaya
   Fourier inversion; Glasserman–Kim Bessel/gamma series) but their finite-arithmetic realisation is not exact and their
   time-inhomogeneous / random-length interface (D4) is not verified. *Investigate:* certified inversion error or the
   Glasserman–Kim series with rigorous truncation, as a separate primitive.
2. **(D3) Target bridges.** Paper 2 needs (i) full-time target/guide equivalence (Assumption 5.3: Lipschitz coefficients,
   global uniform ellipticity, positive smooth transition density, transported Gaussian *upper* bound with certified
   constants). Heston violates global Lipschitz/ellipticity (`√z`, boundary at 0); the guide-side compatibility of Paper 1
   §8 is stated only for the global setting. Paper 2 explicitly says the target identity and the moment certificate are
   *not* established for Heston/CIR/CEV proposals (Cor. 7.5).
3. **Deterministic moment certificate `M_c ≥ E_Q[Ψ(1+log⁺Ψ)²]`.** Requires heat-kernel bounds (Theorem 8.4) — unavailable —
   or a model-specific analytic bound of the guided likelihood moments. Random local coefficient bounds cannot replace it.
4. **Certified guided path.** The Paper 1 verifier on boxes `z ≥ ℓ` needs `D^5` bounds growing like `ℓ^{-9/2}` and the
   block certificate suffers the same enclosure inflation as in the nonlinear benchmark (`docs/CHECKLIST.md`).
5. **Nonexplosion + domain preservation of the *jump* model.** Lyapunov functions for the coupled Heston jump kernel and jump
   maps that keep `V>0` must be supplied; diffusion-only boundary tests do not suffice (Paper 3 App. C.2).
6. **Localization radii.** Paper 3 Prop 7.1 needs a proper gauge `g_D` (e.g. `|x| + 1/dist(x,D^c)`) whose path supremum can
   be certified from bridge records *of the original-domain target bridge* — again requires item 2.

## Suggested route (not implemented)
* Investigate an affine-structure transition/bridge primitive (Broadie–Kaya style) for `(D2)` and the affine jump case,
  independent of the rough-path machinery.
* For a *regularised* Heston (e.g. variance floor / truncated `√z`) all Papers' global assumptions can hold, but the object
  is then a different model and any result must be labelled **regularised/approximate**.
