# FAR²-5M Design decisions

Decisions that are not obvious from the code, with the reasoning. The config dataclasses in
`far2/config.py` are the source of truth for all values; `python -m far2.presets --table`
is the authority on parameter counts.

## v0.2 validity revision (reviewer pass, 2026-08-22)

1. **Slot-recurrent baseline C** (`slot_recurrent_5m`): FAR² topology with zero operators,
   parameter-identical to `far2_5m` (both 4,545,992 as of v0.4). D−C isolates the FAR²
   contribution from the latent-bottleneck advantage; C also joins Phase 0 and the ablation grid.
2. **Family-token leak closed**: the leading F-token is shuffled per episode in training
   AND evaluation (`family_token_mode="random"` default), so `F6 → SPLIT` cannot be
   learned. Telemetry keys off `Sample.family`. `"true"` mode remains as the leak ablation.
3. **`split_label` → `requires_distinction`**: a task condition (this query needs the
   near-identical pair kept distinct), used for curriculum supervision and behavioral
   correlation — never as proof the model internally entangled/disentangled anything.
4. **Binding diagnostics before SPLIT claims**: the harness computes Alignment(slot, entity)
   from final cross-attention (record field `binding`; metric `binding_separation`). If
   separation stays at control rate, SPLIT is generic capacity allocation — a different
   (still reportable) result.
5. **P3/P4 circularity removed**: P3a proves EXPAND/SPLIT mechanics under oracle
   curriculum → P4 runs the signal tournament and writes `runs/selected_signal.json` →
   P3b trains the free controller on the selected signal (`ModelConfig.collision_signal`).
6. **Halting and reconsideration decoupled**: P5a tests RECONSIDER at fixed T=8 (the HALT
   action is masked when `halting != "learned"`), P5b adds learned halting afterward and
   reports honestly against the free convergence rule.
7. **Epistemics are answer-level at 5M**: the supervised mechanism is the answer-level
   4-way head. Per-slot type tags exist behind `slot_epistemic=False` (unsupervised,
   off by default) for later scales.
8. **Compute matching is explicit**: episode records carry inference FLOPs; trainers log
   cumulative training FLOPs; `run_ablations --flop-budget` produces the
   training-FLOP-matched comparison alongside the default step-matched one.
9. **Literature rules softened to flagged hypotheses**: iteration embeddings default OFF
   but exist in both recurrent models (`iter_embedding=True` is a supported ablation);
   the IPT recipe is `ipt_alpha` (0 disables); fp-penalty/JSRR/convex-gate remain the
   pre-registered contingency ladder, not defaults.
10. **Transfer families** (`far2/data/transfer.py`): collision/framing/unknowability
    conditions in unseen surface grammars, excluded from every mix (the sampler refuses
    them), evaluated only by `run_transfer.py` — the headline experiment.
11. **Wording**: H is *immutable across recurrence but differentiable through training* —
    computed once per episode inside the graph, never detached.

## v0.3 causal controls (second reviewer pass, 2026-08-22)

Standard: **if FAR² wins, there should be almost no easy alternative explanation
for why it won.** v0.2 fixed baseline fairness; v0.3 adds the controls that
isolate *which mechanism* caused a win, and *whether decisions (not just
capacity, noise, or auxiliary losses) carry the value*.

1. **Controller policies** (`ModelConfig.repr_policy`): `learned` (default) |
   `random` (EXPAND/SPLIT at rate-matched random times with a *random split
   source* — a control must not aim) | `oracle` (SPLIT forced by
   `requires_distinction` at t=T//2 — the perfect-decision upper bound; it
   consults ground truth at run time, so it is reported as a bound, never a
   win). Non-learned policies keep their controller modules so all three are
   parameter-identical (asserted: `PARAM_IDENTICAL_TO_FULL`). Random-arm rates
   are matched to the trained learned arm's measured firing rates by the
   scripts. Eval-time random draws come from fixed per-iteration seeds so
   control arms replay deterministically.
2. **Necessity-vs-allocation quartet** (`run_phase3b.py`): fixed4 / random /
   fixed16 / adaptive, with the pre-registered reading printed: random closes
   the capacity gap → capacity/noise story; adaptive ≈ fixed16 at ≤70%
   slot-compute while random ≈ fixed4 → intelligent allocation; adaptive >
   fixed16 while random ≈ fixed4 → restructuring beyond capacity.
3. **Episode-level do() interventions** (`run_interventions.py`,
   `forward(..., intervene={"block": {...}, "force": [(op, t)]})`): replay the
   *same episode* (eval is deterministic) with one operator blocked on
   correct-and-used episodes (necessity: flip-to-wrong rate), forced on
   wrong-and-unused condition episodes (sufficiency), and forced on
   correct-no-condition episodes (placebo — a real operator should be
   near-harmless where it isn't needed). P(correct|do(op)) vs P(correct|do(no-op))
   is the causal evidence behind the acceptance gate's "causally improves".
4. **Reset-at-random control** (`reconsider_policy="random"`, rate-matched):
   isolates the learned RECONSIDER trigger from generic state perturbation.
   Plus the thinking-longer control: the P5a no-reset arm is also evaluated at
   T=12 — if plain extra iterations recover as much, direction-change adds
   nothing and we say so.
5. **Representation probes** (`far2/eval/probes.py`, `run_probes.py`): frozen
   ridge probes decode each collision entity's divergent value from (a) the
   slot *bound* to it, (b) the *other* entity's slot, (c) the pooled state.
   Separation evidence = bound ≫ wrong-slot. Complements attention-based
   binding: probes measure content, binding measures where evidence lands.
6. **Goal-capacity control** (`goal_scrambled=True`): the protected-goal
   channel stays (identical params/compute) but carries another episode's
   goal. far2_no_goal vs far2_goal_scrambled separates "protection helps"
   from "an extra clean feature channel helps".
7. **Epistemic-head-only control** (`TrainConfig.epistemic_shuffled`): aux
   labels permuted within batch — same head, same loss pressure, zero
   epistemic content. far2_no_epistemic vs far2_epi_shuffled separates the
   mechanism from aux-loss regularization.
8. **Fine-grained action ablations** (`ModelConfig.disable_actions`): e.g.
   `far2_no_split` keeps EXPAND but never unlocks SPLIT, at identical
   parameters.
9. **Ablation matrix at two tiers** (`run_ablations.py --tier diagnostic`):
   17 arms don't all need 3 full runs immediately — the diagnostic tier
   (1 seed, ¼ steps) triages which comparisons get promoted.
10. **Math, not wording** (`far2/eval/stats.py`, `docs/CRITERIA.md`): every
    gate is an inequality over defined estimators with a pre-registered margin
    AND a significance requirement — superiority/non-inferiority two-proportion
    z-tests, Wilson-bounded single rates, exact binomial vs known baselines,
    exact McNemar for the paired probes. n<20 cells report NOT-TESTABLE and
    fail; they can never pass silently.

## v0.4 exact mathematics (third reviewer pass, 2026-08-22)

The wording-based families could not separate "reasoning depth" from "surface
length", and single-token numbers (N0–N99) allowed lookup-table arithmetic.
v0.4 rebuilds the entire task layer as exact mathematics:

1. **Digit sequences, not number tokens**: numbers are MSD-first digit runs
   (`num_tokens`), so unseen lengths are representable and a 100×100 lookup
   surface is impossible. Answers are zero-padded to a fixed width per surface
   so ANS-slot count never leaks magnitude.
2. **Shared Abacus digit-significance embedding in ALL four models**
   (`e = e_token + e_pos + e_digit_significance`, random per-batch offset at
   train time): FAR² must never win by beating a handicapped arithmetic
   baseline. This is a fairness contract, asserted by tests.
3. **M0–M9 families with exact solvers** (`far2/data/*`): carry_add (difficulty
   = exact carry-chain depth D, independent of digit count L), multiply,
   base_arith (counterfactual bases — train {5,7,10}, test {6,9,11}, base
   declared in-problem), expr_eval (tree depth ⊥ token count via multi-digit
   parenthesized leaves + thin caterpillars), linsys (Fraction-exact RREF;
   DERIVED / UNDERDETERMINED / NO_SOLUTION), symbreak (R = broken pairs = the
   SPLIT condition; d=0 pairs are surface-matched controls), rewrite,
   differentiate, extraneous (sqrt trap — RECONSIDER as real mathematics),
   optimize (protected goal; infeasible-claim trap). Held-out transfer:
   sym_poly, rational_trap, iter_apply, congruence.
4. **Mathematical Generalization Surface**: every record carries (D, R, L);
   the claims are slope patterns — ∂Iter/∂D > 0 with L controlled,
   ∂Slots/∂R > 0 with D, L controlled (`stats.ols2`), plus matched
   counterexamples (32-digit carry-free vs 24-deep chain at identical L).
5. **Exact residual telemetry**: per-iteration probe decodes are scored by the
   family's exact checker (`residual_traj`) — telemetry only, never a loss.

## v0.4 binding substrate + verdict discipline (fourth reviewer pass, 2026-08-22)

A 40-step toy checkpoint showed the probe/gate stack correctly refusing an
untrained model — and exposed that nothing guarantees "slot = concept" on the
standard substrate. The substrate program (R0–R9, see ROADMAP):

1. **R0 probe self-validation** before any claim: planted encodings recovered
   >95%, shuffled labels ≈ chance — else the instrument, not the model, is
   broken and nothing is reported.
2. **R1 location probes** (bound / wrong slot / pooled Z / encoder H anchor /
   nonlinear) with the Case A–D diagnosis: distributed / bottleneck /
   upstream / nonlinear — WHERE the value lives decides what gets fixed.
3. **Competitive binding attention** (`cross_mode="competitive"`): softmax
   over SLOTS per evidence token (Slot-Attention bias) — slots compete for
   evidence instead of all reading everything.
4. **Register slots** (`n_registers=2`): always active, never split sources —
   background information needs somewhere to go that isn't a concept slot.
5. **Binding matrix** `B[e,s] = softmax_s(h_e^T W_B z_s)`: differentiable
   variable→slot ownership; drives collision detection
   (`Collision = Overlap(B_x,B_y) × JSD(P_x,P_y)`, comparing CONCEPTS not
   slot pairs), the SPLIT source, and the directed split
   `z_new = z_src + β·Normalize(proj(h_y − h_x))`.
6. **Binding curriculum** (`bind_nll`, permutation-free soft-assignment CE):
   annealed 0.5 → 0.25 → 0 across stages; zero before any transfer claim.
7. **R8 usage-prerequisite gates before causal claims**: UsageRate ≥ 15%,
   FalseUsage ≤ 5%. "Condition present 500×, operator used 3×" is
   **USAGE-FAILURE** — the controller did not learn the operator — and is
   never conflated with NOT-TESTABLE (population too small). Four-state
   verdicts everywhere: PASS / FAIL / NOT-TESTABLE / USAGE-FAILURE.
8. **R4 strict probe gates** on the binding substrate: bound ≥ chance+20pts,
   bound − wrong ≥ 15pts with exact McNemar — else SPLIT claims are reported
   as capacity allocation.
9. **No 30M scale-up** until the substrate program produces a surviving core
   result at 5M.
10. **Campaign funnel** (`run_campaign.py`): smoke → pilot (killer-result
    chain with stop rules; P5–P7 deferred) → confirm (survivors, evals in the
    thousands) → seeds → grid. Training selective, evaluation large.

## Shared I/O contract

- **One 72-token mathematical vocabulary** for all families
  (`far2/data/vocab.py`): specials + markers + operators + digits D0–D9/DA +
  family tokens F0–F9 + variables X0–X15. Every sample is
  `family-token → context → query → ANS slots`; models predict at ANS
  positions (no autoregression). Answers are fixed-width digit runs (or
  status tokens) so exact match is unambiguous and width leaks nothing.
- Numbers are **digit sequences** (MSD-first, NEG-prefixed, zero-padded);
  `digit_significance` is embedded by every model (Abacus trick, shared).
- Task conditioning = the family token in the input — but it is **shuffled
  per episode** everywhere except the explicit leak ablation, so it carries
  no information; telemetry keys off `Sample.family`.
- Generators are pure functions of `(seed, difficulty)` and each ships an
  independent exact `solve()` used by the pytest ground-truth audit (200
  samples × 14 families), plus optional `residual()` for progress telemetry.

## Stability defaults (from the recurrence-stability literature pass)

- **State update**: `Z ← RMSNorm(Z + a ⊙ G ⊙ ΔZ)` — the spec's additive gated form **plus an RMSNorm
  outside the residual**. Pre-LN alone → state norm grows linearly with iterations; external norm is
  what keeps T ≫ T_train bounded (Geiping n_o; STARS norm-placement findings). Norm applied only to
  active slots so inactive slots stay exactly at their seeds.
- **Gate**: bias init −2 (E[G] ≈ 0.12, near-identity start with live gradient), weight init ~1e-3.
  `gate_style="convex"` ((1−G)Z + GΔZ, bounded by construction) is the Phase-2 contingency.
- **Recall**: evidence H is cross-attended **every iteration** (FAR²) / input embeddings re-added every
  iteration (recurrent baseline, `recall=True`). The literature's dominant depth-extrapolation lever.
- **No iteration embeddings** by default anywhere — they break weight-sharing extrapolation.
- **IPT loss**: (1−α)·CE(final) + α·CE(one random intermediate iterate vs the same final target),
  α=0.5. Never weight later iterates more; never supervise step-specific targets.
- **State-norm penalty** 0.01·meanₜRMS²(Zₜ): scale-invariant readouts have zero radial gradient —
  the loss literally cannot see state magnitude without this.
- Optimizer: AdamW(0.9, 0.95), lr 3e-4 warmup+cosine, wd 0.01 (none on norms/bias/gates/embeddings/
  seeds; gates in their own param group), grad clip 0.5, **weight EMA 0.999 (eval on EMA)**, full BPTT.
- Contingency ladder for the 8>16>32 failure signal (all implemented, default off):
  heavy-tailed T sampling → terminal fixed-point penalty ‖Z_T−Z_{T−1}‖² (note: literature conflict —
  terminal-only helped in one study, trajectory-wide smoothing hurt in another; we ablate) →
  JSRR λ=0.1 (1-step JVP power iteration) → convex gate.

## Slot / controller defaults (from the dynamic-slot literature pass)

- Inactive slots = **learned seeds** (not zeros) + small jitter at activation; active mask multiplies
  the **update**, not just attention logits; masked attention uses finite fill (−1e4), never −inf.
- SPLIT clone: `z_clone = z_src + β·d_nearest(z_src)·v̂`, β=0.2; **clone only, never the source**;
  noise is mandatory (identical copies = permanent saddle point).
- Controller training: supervised oracle warmup (generator labels) → free straight-through
  Gumbel-softmax, τ 1.0 → 0.5, small decaying entropy bonus (0.01) → cost penalties ramp last.
  Diagnostic: soft-path vs hard-path loss gap (ST discretization gap hits SPLIT hardest; ReinMax is
  the designated fallback estimator).
- Collision monitor C = cos(zᵢ,zⱼ) · JSD(pᵢ,pⱼ) where pᵢ = per-slot answer distribution through the
  shared head. This targets the documented gap in adaptive-slot work (their reconstruction-error
  signals catch redundant slots, not entangled ones). Phase 4 benchmarks it against alternatives
  (incl. attention-overlap soft-IoU) instead of assuming it.
- Halting: learned CONTINUE/RECONSIDER/HALT per spec, with HALT trained TRM-style (BCE of a
  correctness head), and the free convergence rule r_t = ‖ΔZ‖∞/(‖Z‖∞+ε) < 0.1 as the honest control
  (the literature says learned halting rarely beats it).

## Parameter budgets (v0.4 counts; printed at startup, asserted 4.2–5.8M in tests)

- `transformer_5m` 4,917,576 · `recurrent_5m` 4,561,480
- `slot_recurrent_5m` = `far2_5m` = `slot_recurrent_bind` (14+2 slots) = 4,545,992
- `far2_full` 4,775,763 · `far2_bind` = `far2_bind_random_expand` = `far2_bind_oracle` = 4,980,563
- `python -m far2.presets --table` prints the per-module ledger; the
  `PARAM_IDENTICAL_TO_*` tuples are asserted by tests.

## Known deviations from the original spec (all deliberate, all flagged)

1. RMSNorm outside the residual added to the spec's update equation (stability; see above).
2. Task-specific decoders replaced by family-token conditioning (fairness across models).
3. Learned halting supplemented by a convergence-rule control in Phase 5 (evidence-backed control).
4. The spec's uniform T grid {2,3,5,6,8} runs first; heavy-tail sampling is a contingency, not the default.
