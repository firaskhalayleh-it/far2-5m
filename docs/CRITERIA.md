# Pre-registered decision rules (v0.4)

Every gate is an inequality over defined estimators — margin **and**
significance — never wording. Implementation: `far2/eval/stats.py` (exact
binomial, Wilson intervals, pooled two-proportion z, exact McNemar, two-
regressor OLS); the code is the authority and `tests/test_stats.py` pins it
to hand-computed values. Families are the v0.4 mathematical set (M0–M9 +
held-out transfer); every number a gate consumes comes from an exact solver.

## Conventions (fixed project-wide)

- Episode outcome `correct(e) ∈ {0,1}`; estimator `p̂ = k/n` over the stated
  episode set. All (k, n) come from `far2.eval.metrics` count functions.
- **α = 0.05, one-sided**, per pre-registered comparison. No harvesting:
  the comparisons below are the confirmatory set; anything else is exploratory.
- **CI**: Wilson score interval at z = 1.96.
- **Superiority** (arm 1 better than arm 2 by margin m):
  `PASS ⇔ (p̂₁ − p̂₂ ≥ m) ∧ (p < α)` with
  `z = (p̂₁ − p̂₂)/SE_pooled`, `SE_pooled = √(p̄(1−p̄)(1/n₁+1/n₂))`,
  `p = 1 − Φ(z)`.
- **Non-inferiority** (arm 1 not worse than arm 2 by more than m):
  H₀: `p₁ ≤ p₂ − m`; `z = (p̂₁ − p̂₂ + m)/SE_Wald`; `PASS ⇔ p < α`.
- **Single rate, lower-bounded**: `PASS ⇔ (p̂ ≥ point) ∧ (Wilson_lo ≥ floor)`.
- **Single rate, upper-bounded**: `PASS ⇔ (p̂ ≤ point) ∧ (Wilson_hi ≤ ceiling)`.
- **Known-baseline rate** (probe vs chance p₀): exact binomial,
  `PASS ⇔ (p̂ ≥ p₀ + m) ∧ (P(X ≥ k | n, p₀) < α)`.
- **Paired comparison** (two probes on the same rows): exact McNemar,
  `b = #(A right, B wrong)`, `c = #(A wrong, B right)`,
  `p = P(X ≥ b | X~Bin(b+c, ½))`.
- **Surface regression** (`stats.ols2`): y on two standardized regressors;
  the claim pattern is `β₁ > 0 significant (p₁ < α)` while x₂ contributes
  nothing (`p₂ ≥ α` or `β₂ ≤ 0.5·β₁`).
- **N_MIN = 20**: any cell with fewer eligible episodes is NOT-TESTABLE and
  the gate FAILS with that verdict (never a silent pass).
- **Verdict vocabulary** (operators): PASS | FAIL | NOT-TESTABLE (condition
  population too small) | USAGE-FAILURE (the controller did not learn to use
  the operator). The last two are different findings and are never lumped.
- Deterministic quantities (parameter counts, FLOP ratios, mean active slots)
  are compared directly — no test needed, the inequality is the rule.

## Campaign funnel (how gates are consumed)

Smoke (~300 steps) → pilot (~2.5–3k steps, seed 0, large evals) → confirm
(5–10k steps, thousands of eval episodes, survivors only) → seeds (3–5,
important survivors only) → full ablation grid (only after core gates pass).
Pilot gate outcomes decide *promotion*, not claims; only confirm-stage runs
feed the acceptance gate. Training is selective; evaluation is large
(synthetic generation is free — decisive evals use thousands of episodes at
confirm stage).

## Phase gates as inequalities

Notation: `p̂_X(S)` = accuracy of arm X on episode set S; `dq` = distinction
queries (requires_distinction = 1, family symbreak unless stated);
`rec(X)` = recovery counts (metrics.recovery_counts, family extraneous or
rational_trap); `m` = margin.

### P0 — baselines (transformer / recurrent / slot_recurrent)

Easy-bucket accuracy ≥ 95% per core family (trainability, not a comparison);
params ∈ [4.2M, 5.8M]; FLOPs logged per episode. All deterministic checks.

### P1 — stable recurrence (far2, operators off; T=4 vs T=1)

1. Training does not diverge (all losses finite, < 10⁴).
2. State RMS ∈ [0.5, 8] with |slope| < 0.05 across iterations.
3. Overall: `p̂_T4 ≥ p̂_T1 − 0.01`.
4. Hard bucket (difficulty ≥ TRAIN_DMAX): `p̂_T4 ≥ p̂_T1 + 0.05`.
5. Readout at T ∈ {6, 8}: `p̂ ≥ p̂_T4 − 0.02` (no collapse).
STOP RULE: if this gate fails, nothing downstream is built.

### P2 — mathematical depth extrapolation (trained ≤ frontier, eval beyond)

1. In-distribution at T=2: `p̂ ≥ 0.60`.
2. OOD monotonicity: `p̂(T=4) ≥ p̂(T=2) + 0.03`, `p̂(T=8) ≥ p̂(T=4) + 0.03`.
3. Stability: `p̂(T=16) ≥ p̂(T=8) − 0.02`.
4. Length-matched counterexample (32 digits, D=0 vs D=24, identical L):
   `p̂_D0 ≥ p̂_D24 + 0.05` or `p̂_D0 ≥ 0.90` — length alone must not be the cost.
Failure signal (not a gate): `p̂(T=8) − p̂(T=32) > 0.10` triggers the
pre-registered remedy ladder (heavy-tail T, fp_penalty, JSRR, convex gate).

### P3a (R2) — oracle-necessity quartet (fixed4 / fixed16 / oracle-EXPAND /
oracle-SPLIT), binding substrate

1. The benchmark REQUIRES representation: superiority fixed16 vs fixed4 on
   dq, m = 0.05. STOP RULE: if this fails, a controller is CORRECT never to
   split — fix the benchmark before training one.
2. Superiority oracle-SPLIT vs fixed4 on dq, m = 0.05.
3. Non-inferiority oracle-SPLIT vs fixed16 on dq, m = 0.02.
4. oracle-SPLIT vs oracle-EXPAND contrast reported (targeting vs capacity).
5. Binding diagnostics present (n > 0) — claim-prerequisite for SPLIT.
Oracle arms consult ground truth at run time: upper bounds, never wins.

### P3b (R7) — necessity-vs-allocation quartet (F4, RND, F16, ADP)

- Capacity gap: `Δ_cap = p̂_F16(dq) − p̂_F4(dq)`; quartet informative only if
  `Δ_cap > 0.02`.
- Random-closure index: `R = (p̂_RND(dq) − p̂_F4(dq)) / Δ_cap` (RND is
  rate-matched to ADP's measured firing rates, split source random).
- Compute ratio: `ρ = (slots·iters)_ADP / (slots·iters)_F16`.
- Gates:
  1. Non-inferiority ADP vs F16 on symbreak, m = 0.01.
  2. `ρ ≤ 0.70`.
  3. Superiority ADP vs RND on dq, m = 0.03.
  4. Split precision `tp/n_did`: point ≥ 0.60, CI floor 0.50.
  5. Split recall `tp/n_need`: point ≥ 0.50, CI floor 0.40.
  6. Mean active slots ∈ [3.5, 15.5] (no collapse).
  7. Surface: active_slots on (R, L) — `p_R < α` and (`p_L ≥ α` or
     `β_L ≤ 0.5·β_R`). "More equations → more slots" would load on L.
  8. Binding separation after split > separation on control queries (m = 0).
- Pre-registered reading (printed with R and ρ):
  `R ≥ 0.6` → capacity/noise; `NI(ADP,F16) ∧ ρ ≤ 0.7 ∧ R ≤ 0.4` → allocation;
  `p̂_ADP − p̂_F16 ≥ 0.03 ∧ R ≤ 0.4` → restructuring beyond capacity.

### P5a — reconsideration on extraneous roots (CTL, RST(random), REC; CTL@T12)

1. Superiority rec(REC) vs rec(CTL), m = 0.05 (identical compute).
2. Superiority rec(REC) vs rec(RST), m = 0.03 (decision vs perturbation).
3. Non-inferiority rec(REC@T8) vs rec(CTL@T12), m = 0.01 (direction vs
   duration).
4. Learned reset rate ∈ [0.05, 6] events/episode.
5. Non-inferiority of overall accuracy REC vs CTL, m = 0.02.

### P6 — epistemic state on linear systems (DERIVED / UNDERDETERMINED /
NO_SOLUTION)

1. Unsupported assertion rate: point ≤ 0.10, ceiling 0.20.
2. Over-abstention: point ≤ 0.10, ceiling 0.20.
3. Answerable accuracy: non-inferiority vs control, m = 0.02.

### P7 — protected goal on constrained optimization

1. Infeasible-claim rate: `viol_PROT ≤ 0.5 · viol_CTL` AND two-proportion
   contrast p < α.
2. Accuracy non-inferiority vs control, m = 0.02.

### Interventions — per operator op ∈ {SPLIT, EXPAND, RECONSIDER}

**R8 usage prerequisite (before any do()):** over the natural pool,
`UsageRate = P(used | cond)` and `FalseUsage = P(used | ¬cond)`.

1. UsageRate: point ≥ 0.15, Wilson floor 0.10. Below → **USAGE-FAILURE**
   ("condition present 500×, used 3×" is a finding about the controller,
   not an unavailable test). `n_cond < N_MIN` → **NOT-TESTABLE**.
2. FalseUsage: point ≤ 0.05, Wilson ceiling 0.15.

**R9 do() (only if R8 passed).** Pools (replay is deterministic, so every
flip is caused by the do()):
`E⁺ = {e : correct ∧ used(op)}`, `E⁻ = {e : ¬correct ∧ ¬used(op) ∧ cond(e)}`,
`E⁰ = {e : correct ∧ ¬cond(e)}`.

3. Necessity `p̂_flip(E⁺, block op)`: point ≥ 0.10, Wilson floor 0.05.
4. Placebo `p̂_flip(E⁰, force op)`: point ≤ 0.15, Wilson ceiling 0.30.
5. Sufficiency `p̂_flip(E⁻, force op)`: reported with CI, not gated (a single
   forced action at t = T/2 is a crude do()).
6. Reported contrast: two-proportion z of necessity vs placebo flip rates.

### Probes — R0 → R1 → R4 ladder (symbreak pair values; chance p₀ = 1/10)

**R0 self-validation (before any claim):** planted synthetic encodings
recovered > 95% AND shuffled-label accuracy < p₀ + 0.15 — else the
instrument is broken and nothing about the model is reported.

**R1 location probes** on identical paired rows (episode-level train/test
split): bound slot (linear + MLP), wrong slot, pooled Z, encoder H at the
variable anchor. Diagnosis is the pattern: Case A (H, pooled high; slots ~
chance) = distributed; Case B (H high, pooled low) = bottleneck; Case C
(H low) = upstream; Case D (linear low, MLP high) = present, not linear.

**Baseline gates** (standard substrate): bound vs p₀ exact binomial
m = 0.10; separation `(p̂_bound − p̂_wrong ≥ 0.10) ∧ (McNemar p < α)`.

**R4 strict gates** (binding substrate, `--strict`): bound vs p₀ m = **0.20**;
separation `(p̂_bound − p̂_wrong ≥ 0.15) ∧ (McNemar p < α)`. If these fail,
SPLIT claims must be reported as capacity allocation, not concept separation.

### Transfer (held-out families; the headline)

1. Spontaneous split tracks the unseen condition (sym_poly): superiority of
   `tp/n_need` vs `fp/n_ctrl`, m = 0.10. No family token, no oracle.
2. Recovery transfers (rational_trap): superiority rec(FAR²) vs
   rec(slot-recurrent C), m = 0.05. Without a trained control: NOT-TESTABLE
   (fail).
3. Unsupported assertions on unseen unknowables (congruence): point ≤ 0.20,
   ceiling 0.35.
4. Over-abstention on unseen answerables: point ≤ 0.20, ceiling 0.35.
5. iter_apply depth curve reported (recurrence transfer, exploratory).

### Acceptance gate (5M, v0.4)

Three claims are MANDATORY, each under the rules above at confirm stage:

1. **Computation adaptation**: P2 gates 1–3 (iterations buy OOD depth) plus
   the length-matched counterexample (gate 4).
2. **Representation adaptation**: P3b gates 1–3 + 7 (adaptive beats
   rate-matched random, tracks R not L, at ≤ 70% slot-compute).
3. **Cross-domain operator transfer**: Transfer gates 1–2.

Plus **≥ 2 of 6**: P5a reconsideration; P6 epistemics; P7 protected goal;
learned halting (P5b); interventions full causal chain (R8 PASS + necessity +
placebo per op); R4 probe separation on the binding substrate. Every
mechanism claimed must also show: training-level ablation effect + its
random-policy control loses + (SPLIT claims) binding separation significant.
The transfer result is reported with seed error bars either way.

## Multiple comparisons

The confirmatory set is small and pre-registered per phase (≤ 8 comparisons);
we report per-comparison p at α = 0.05 AND flag in the dashboard any phase
where the number of passing gates would be expected under global null
(binomial sanity line). Exploratory slices (per-family tables, per-difficulty
curves, iter_apply depth curve) carry CIs but no pass/fail claims.
