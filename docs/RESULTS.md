# FAR²-5M current result snapshot

Date: 2026-08-29

This file records measured results, separately from the pre-registered method
in `CRITERIA.md`. It is not a claim that the full campaign passed.

## v0.4.52 capability contract — IMPLEMENTED, NOT RUN

The next revision now separates fresh-ID symbreak R∈{1,2,4} from OOD R=6,
supports paired Z0/T1/T4/T8 readout with zero recurrent FLOPs at Z0, adds the
parameter-identical slot-recurrent-bind control to the foundation curve, and
uses a staged easy→rules→real-task curriculum while keeping the selected arm
final-answer-only, operator-free, and without oracle/entity/binding labels.
Transfer now rejects a checkpoint trained on held-out families and verifies
that learned tensors are unchanged before and after evaluation.

No v0.4.52 training or transfer experiment has been launched. These changes
strengthen trainability and capability measurement; they are not evidence that
model capability increased. The measured boundary below remains authoritative
until a complete v0.4.52 curve exists.

## v0.4.51 foundation repair — plumbing PASS; recurrent reasoning NOT-SHOWN

Fresh isolated evidence from `runs_v0451/smoke_final` (2026-08-23):

- Active FAR2 slots are initialized through the existing encoder-to-workspace
  cross-attention. The decoder remains workspace-only; it receives no H bypass.
- Intermediate supervision now decodes every pre-final recurrent state. The run
  contract records its mode, weight, forward cycles, optimizer updates, unique
  examples, supervised intermediate states, and the extra decoder FLOPs.
- Binding-mode SPLIT is now a semantic bifurcation: parent plus two contextual
  entity states produce complementary children, a 2x2 Sinkhorn assignment
  selects exclusive owners, and that ownership persists for the episode.
- A seven-layer `transformer_bind_match` control has **5,707,336 parameters**,
  versus **5,765,853** for FAR2-bind, a **1.02%** gap.

Isolated acceptance:

| v0.4.51 layer | Evidence | Verdict |
|---|---:|---|
| Mechanical representation/operator laws | 6/6 | PASS |
| Learned foundation gates | 7/7 | PASS |
| Gradient reachability | encoder, initializer, core, binding, decoder all non-zero | PASS |
| Direct two-entity extraction | 100% entity accuracy | PASS |
| Workspace bind/read | 100% exact at T=1/4/8 | PASS |
| Input-conditioned causality | 100% correct warm state vs 18.0% shuffled | PASS |
| Learned-seed control | 46.1% T1, 55.5% T4, 64.8% T8 | DIAGNOSTIC |
| Full regression suite | 148/148 | PASS |

The controlled task is intentionally trivial. It proves that every learned
layer can carry, retain, and decode entity information; it does not prove the
symbreak benchmark, adaptive allocation, transfer, or general reasoning.

Audited 600-update full-stack smoke (19,200 unique examples per arm, 96 held-out
one-digit-multiplication evaluations):

| Model | Parameters | Exact | Token | Train FLOPs | Intermediate states supervised |
|---|---:|---:|---:|---:|---:|
| Transformer tiny | 144,776 | 81.25% | 90.63% | 0.378T | 0 |
| Recurrent tiny | 111,496 | 26.04% | 58.85% | 0.981T | 0 |
| FAR2 core tiny | 253,384 | 100% | 100% | 0.702T | 57,600 |
| FAR2 full tiny | 265,731 | 100% | 100% | 0.702T | 57,600 |
| FAR2 binding tiny | 299,917 | 100% | 100% | 0.769T | 57,600 |

This table is **quarantined from architecture comparison**. The tiny models are
not parameter-matched, FAR2 used more training FLOPs than the Transformer, and
the shortcut audit below shows that the easy answer is already mostly present
at initialization and complete after one cycle.

### Mandatory shortcut audit — recurrent-reasoning claim FAIL

The same 96 held-out episodes were replayed on every FAR2 checkpoint. `Z0`
decodes immediately after input-conditioned initialization, before the
recurrent loop. Learned and shuffled modes reuse identical trained weights.

| Checkpoint | Warm Z0 | Warm T1 | Warm T4 | Learned-seed T4 | Shuffled T4 | Reading |
|---|---:|---:|---:|---:|---:|---|
| FAR2 core tiny | 77.08% | 100% | 100% | 1.04% | 1.04% | initialization-dominant |
| FAR2 full tiny | 77.08% | 100% | 100% | 1.04% | 1.04% | initialization-dominant |
| FAR2 binding tiny | 62.50% | 100% | 100% | 3.12% | 1.04% | initialization-dominant |

All three cycle trajectories are `[100%, 100%, 100%, 100%]`. Therefore the
smoke proves an honest input-to-workspace-to-decoder route, but it does **not**
show recurrent reasoning. There is no old contextual-H decoder bypass; the new
shortcut is architectural task migration into `BindInit`/the encoder. That is
acceptable for a wiring smoke and disqualifying for a recurrence claim.
Evidence: `runs_v0451/smoke_shortcut_audit.json`.

### 500-update supervision/initialization ablation

All arms use the same FAR2-core parameters, examples, seed, batch, and T=4.

| Arm | T1 exact | T4 exact | Train FLOPs | Decision |
|---|---:|---:|---:|---|
| learned seed + final-only | 13.54% | 22.92% | 0.447T | control |
| learned seed + all-cycle | 26.04% | 27.08% | 0.586T | insufficient |
| warm-start + final-only | 47.92% | **90.63%** | **0.447T** | **selected** |
| warm-start + all-cycle | 89.58% | 97.92% | 0.586T | redundant at smoke gate |

The decision law selects **warm-start plus final-only loss**, the simplest arm
that passes. Its cycle trajectory `[47.92%, 56.25%, 84.38%, 90.63%]` also shows
that recurrence contributes at 500 updates, unlike the saturated 600-update
all-cycle checkpoint. All-cycle supervision remains implemented as an ablation,
not the selected 4,000-update recipe. Applying it to the recurrent baseline at
weight 0.5 reduced that baseline from 26.04% to 19.79%, confirming that it is
not a universally beneficial optimization.

**Current boundary:** v0.4.51 repairs plumbing and SPLIT semantics. It does not
beat the Transformer and does not yet demonstrate recurrent reasoning on a task
that requires it. The v0.4.5 fixed14/P3 result remains the last real-task
evidence and remains NOT-READY. The next admissible pilot evidence is the
v0.4.52 checkpointed 4,000-update curve on the P3 mixture: warm versus
learned-seed and all-cycle diagnostics, the parameter-identical
slot-recurrent-bind control, 4.9M and 5.7M Transformers, paired Z0/T1/T4/T8,
fresh-ID separated from OOD, final-answer loss on the selected arm, and no
binding auxiliary labels.

### Real-task curve early stop at 1,000 updates — learning, NOT-READY

Per user instruction, the first arm (`far2_fixed14_warm`) was stopped after its
1,000-update checkpoint and audited before any seed-only or Transformer curve
arm started. Contract: 5,765,853 parameters, batch 128, T=8, final-answer loss
only, no binding auxiliary labels, 128,000 unique examples, 462.260T analytical
training FLOPs, and zero intermediate-supervised states.

Paired distinction-query results across 805 episodes / 6,440 target tokens:

| Update | Z0 token | T1 token | T8 token | T8 exact |
|---:|---:|---:|---:|---:|
| 250 | 5.17% | 5.37% | 6.88% | 0.0% |
| 500 | 12.44% | 12.69% | 13.70% | 0.0% |
| 1,000 | 22.90% | 23.45% | 25.02% | 0.0% |

At 1,000 updates, replaying identical weights with learned seeds gives 13.04%
token accuracy and shuffled input initialization gives 9.98%. Warm initialization
therefore carries useful real-task information, but recurrence contributes only
1.57 points from T1 to T8 (2.11 points from Z0 to T8), far below the planned
10-point useful-recurrence diagnostic. Exact accuracy remains at the boundary
and fixed14 readiness remains **0/2** versus the 20% exact / 75% token gates.

Verdict: the P3 task curve is rising, so the substrate is not flatlined, but
the evidence remains initialization-dominant and does not yet establish
recurrent reasoning. No comparison with a Transformer is available because the
stop occurred before either Transformer arm. Evidence:
`runs_v0451/foundation_curve_seed0/early_stop_1000.json`.

## Architecture and budget

- `far2_bind` and `slot_recurrent_bind`: **5,765,853 parameters each**.
- The repaired P3 binding quartet is parameter-identical: fixed8, fixed14,
  oracle-EXPAND, and oracle-SPLIT each instantiate 5,765,853 parameters.
- The contextual entity path pools every rendered occurrence of a variable,
  reads immutable encoder state H once, and supplies entity state E to the
  binding matrix. It leaves the general recurrent content-attention path intact.
- P4 separates representation demand (`entity_relation`) from workspace
  localization (`binding_overlap` and their product, `bind`). Candidate
  relations come from the rendered clause structure, never the declared probe
  pair or `requires_distinction` label.
- The output decoder receives syntax-only query identity and answer rank; it
  can obtain mathematical content only by reading the recurrent workspace.
- The model remains below 6M. Further expansion is not currently justified by
  the evidence.

## v0.4.5 historical benchmark state — learning NOT-READY

Fresh v0.4.5 evidence (2026-08-22), after removing contextual ANS states from
the FAR2 decoder and changing the oracle from one binary action to one action
per broken pair:

- Mandatory pre-smoke representation preflight: **5/5 PASS** — syntax-only
  output queries, workspace causal sensitivity, binding compatibility/collision,
  full oracle demand service, and free-policy oracle-metadata invariance.
- Fresh 600-step smoke: Transformer 81%, recurrent 26%, FAR2 core/full 39%,
  FAR2 binding 42% exact on one-digit multiplication (modal baseline 7%). The
  former FAR2 100% is superseded; the new result is lower but architecturally
  honest because answers can no longer retain contextual H through Qa.
- The redesigned 150-step P3 diagnostic instantiated **5,765,853 parameters in
  every arm**. Oracle SPLIT executed exactly R actions: 1/2/4/6 on demand and
  zero on matched controls; active slots were 10+R including two registers.
- Mathematical competence remains **NOT-READY**: fixed8/fixed14/oracle EXPAND/
  oracle SPLIT all had 0% distinction-query exact accuracy and 10.9-12.0%
  token accuracy. This blocks capacity/operator comparisons rather than
  declaring them equal at an exact-match boundary.
- Binding separation on the short oracle-SPLIT run improved to 68.1%, but no
  causal representation claim is allowed without task competence.

## Transformer comparison

Two current diagnostic comparisons exist: the fresh smoke below and the
paired 150-step P3 foundation run. Neither is a scientific win: the tiny smoke
presets are not parameter-matched, and 150 P3 steps leave both models at the
exact-match boundary.

| Current smoke model | Parameters | Exact / 96 | Token accuracy | Exact vs Transformer | Smoke gates |
|---|---:|---:|---:|---:|---:|
| Transformer tiny | 144,776 | **81.25%** | **90.63%** | reference | 2/2 PASS |
| Recurrent tiny | 111,496 | 26.04% | 58.85% | -55.21pts | 2/2 PASS |
| FAR2 core tiny | 253,384 | 38.54% | 63.54% | -42.71pts | 2/2 PASS |
| FAR2 full tiny | 265,731 | 38.54% | 63.54% | -42.71pts | 2/2 PASS |
| FAR2 binding tiny | 299,917 | 41.67% | 65.10% | -39.58pts | 2/2 PASS |

At the full budget, `transformer_5m` has 4,917,576 parameters, FAR2 core has
4,709,832 (-4.2%), FAR2 full has 4,939,603 (+0.4%), and FAR2 binding has
5,765,853 (+17.2%). The paired 150-step P3 diagnostic (69 distinction
episodes per arm) produced:

| P3 foundation diagnostic | Exact | Token | Mean inference FLOPs | Readiness signals |
|---|---:|---:|---:|---:|
| Transformer reference | 0.0% | **12.68%** | 1.977B | 0/2 |
| FAR2 binding fixed14 | 0.0% | 11.96% | **1.123B (0.57x)** | 0/2 |

Both are NOT-READY at this short budget. Transformer is +0.72 points on token
accuracy; FAR2 uses 43% fewer analytical inference FLOPs. Neither difference
is a capability claim. P3a now makes this paired comparison automatically and
reports it before operator arms can run. No current full-budget accuracy
comparison is available.

The scientific comparison must be read as **A vs C vs D**: Transformer A vs
FAR2 D measures the overall architecture, while parameter-identical
slot-recurrent C vs FAR2 D isolates the adaptive operators. A direct A-vs-D
win alone cannot prove that SPLIT or autonomy caused the gain.

## Versioned layer acceptance matrix

Pass rates use only declared, testable checks as their denominator. `N/T`
means the prerequisite failed, so a pass percentage would be misleading;
`NOT-RUN` means the current architecture has no evidence for that layer yet.

| Layer | v0.4.4 | v0.4.5 evidence | v0.4.5 pass rate | Verdict |
|---|---|---|---:|---|
| Query/full-demand tensor contract | No standalone law | Query variables and all R broken pairs serialized | 1/1 (100%) | PASS |
| Workspace-only answer route | Contextual ANS state could bypass Z | Syntax-only query; logits change when Z changes | 1/1 (100%) | PASS |
| Collision semantics | Learned signal gate failed; declared-pair probe was non-selectable | Synthetic incompatible collision detected; compatible sharing rejected | 1/1 (100%) mechanical | PASS-MECHANICAL |
| Oracle SPLIT execution | One split regardless of R | R=1/2/4/6 needs get 1/2/4/6 splits; all matched controls get 0 | 8/8 (100%) | PASS |
| Free-policy oracle isolation | Covered indirectly | Perturbing oracle pairs/count changes neither logits, usage, nor collision telemetry | 3/3 (100%) | PASS |
| P3 arm identity/capacity law | Equal parameters, but fixed4/fixed16 did not represent the task's natural floor | All arms 5,765,853 params; fixed8 floor and fixed14 ceiling | 4/4 (100%) | PASS-CONTRACT |
| Software regression suite | 147/147 | 147/147 | 147/147 (100%) | PASS |
| Easy-task smoke learning | 5/5 above modal; FAR2 100% used the invalid answer shortcut | 5/5 above 7% modal; exact: Transformer 81, recurrent 26, FAR2 core/full 39, bind 42 | 5/5 (100%) wiring | PASS-SMOKE |
| Direct Transformer foundation comparison | No matched P3 reference | Same task/budget diagnostic recorded: both 0% exact; Transformer 12.68% vs FAR2 11.96% token; FAR2 0.57x FLOPs | 1/1 diagnostic | PASS-WIRING / NOT-READY |
| Fixed14 mathematical foundation | Exact-match boundary at 0% | Exact 0% and token about 12%, below 20%/75% readiness thresholds | 0/2 (0%) | **FAIL / NOT-READY** |
| Oracle/capacity benefit | Comparisons saturated at 0% exact | Blocked by the fixed14 readiness failure | N/T | NOT-TESTABLE |
| Strict learned binding/localization (R4) | Strict gates failed | 68.1% separation is diagnostic only; current strict probe not run | — | NOT-RUN |
| Runtime detector selection (P4) | 0/1 AUROC gate | Requires a competent current P3a checkpoint | — | NOT-RUN |
| Autonomous allocation (P3b) | Boundary-safe comparison was NOT-TESTABLE | Hard-blocked until P3a and P4 pass | — | NOT-RUN |
| Causal interventions and transfer | Historical checkpoints belonged to the superseded substrate | Must be rerun only after a current P3b survivor | — | NOT-RUN |

Current roll-up: the isolated representation substrate is **5/5 (100%)
mechanically valid**, the regression suite is **147/147**, and the wiring smoke
is **5/5 above the modal baseline**. The learned P3 foundation is **0/2 ready**,
so there is currently **no eligible pass rate** for representation benefit,
autonomy, interventions, or transfer.

## Superseded v0.4.4 standalone smoke — historical only

The prior 600-step CPU smoke on 96 one-digit multiplication evaluations gave
Transformer 81%, recurrent 26%, and FAR2 core/full/bind 100% exact. That result
used the superseded contextual answer-state path. It is retained only as
historical context and must not be used as evidence for the current
architecture.

The v0.4.4 diagnostic below is retained as historical evidence from the
superseded contextual-answer/one-split/fixed4 design.

## Superseded v0.4.4 parameter-identical representation mini — FAIL / NOT-TESTABLE

Configuration: seed 0, 250 steps per arm, batch 128, EMA 0.99, CPU, 160
episodes per requested evaluation bucket. This is a diagnostic budget, not the
pilot or confirm budget.

P3a distinction-query exact accuracy:

| Arm | Exact |
|---|---:|
| fixed4 | 0.0% |
| fixed16 | 0.0% |
| oracle-EXPAND | 0.0% |
| oracle-SPLIT | 0.0% |

- `fixed16 > fixed4 + 5 points`: **FAIL**.
- `oracle-SPLIT > fixed4 + 5 points`: **FAIL**.
- `oracle-SPLIT` non-inferior to fixed16: **NOT-TESTABLE** at the shared zero
  boundary (0/315 vs 0/315), not a pass.
- Oracle split selectivity: 100% when required, 0% on controls; this validates
  execution of the upper-bound policy, not task benefit.
- Binding separation when required: 55.6%; task competence is still absent.

P4 blocked pre-action tournament:

- Best eligible signal: `C`, AUROC **0.542**; required **> 0.55** — **FAIL**.
- `entity_relation`: 0.478; `binding_overlap`: 0.456; `bind`: 0.475.
- No signal selection was written. Production P3b must refuse to train without
  a successful selection (or an explicit user-authorized override).

R0/R1/R4 representation probes:

- Probe self-check: planted 100.0%, shuffled 7.5%, chance 10% — **PASS**.
- Oracle-split: E-head 21.5%, E post-hoc linear 11.8%, bound 13.5%, wrong-slot
  13.5%, separation 0.0% — **R4 FAIL**.
- Fixed16 control: E-head 23.2%, E post-hoc linear 10.3%, bound 13.8%,
  wrong-slot 12.9%, separation 0.9% — no localization advantage.
- Diagnosis: upstream entity learning is improving above chance but remains
  below the strict threshold; the workspace does not localize pair values.

## Verification and decision

- Full repository suite: **147/147 tests passed**.
- `git diff --check`: clean.
- Older binding checkpoints missing the contextual entity modules are rejected
  instead of silently loading those modules with random weights.
- Short diagnostic scripts use EMA 0.99 below 1,000 steps; confirm-scale runs
  retain EMA 0.999, with the choice serialized in every run contract.

**Decision:** the implementation and smoke wiring pass, but adaptive
representation does not. Do not launch P3b, claim that all phases work, or use
the 100% smoke score as evidence against the Transformer. The next authorized
step should be a properly budgeted pilot of the repaired P3a substrate; scale
to 6M only if that run shows a measured entity-capacity bottleneck.
