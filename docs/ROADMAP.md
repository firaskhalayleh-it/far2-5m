# FAR²-5M Roadmap — phases & gates (v0.4, exact-mathematics revision)

> **Prove the cognition mechanism works before proving language capability —
> on exact mathematics, where every answer has a ground truth and every
> wrong answer is a measurable residual.**

Full blueprint (architecture ledgers, operator specs, data samples, metrics):
the published "FAR²-5M Blueprint" artifact + `docs/DESIGN.md`.

Each phase script prints its gate table (PASS/FAIL) and exits nonzero on failure.
**If a gate fails: stop. The failure becomes the research problem.**
Every gate is a pre-registered statistical decision rule — margin AND
significance (α=0.05 one-sided; Wilson CIs; exact binomial/McNemar;
two-proportion z; two-regressor OLS for surface slopes; cells with n<20 are
NOT-TESTABLE, never a pass). Operator verdicts use the four-state vocabulary
PASS / FAIL / NOT-TESTABLE / USAGE-FAILURE. The formal inequalities live in
`docs/CRITERIA.md`; `far2/eval/stats.py` implements them.

The four baselines (all share the Abacus digit-significance embedding — no
model is handicapped by numeric representation):

- **A** `transformer_5m` — no recurrence, no state
- **B** `recurrent_5m` — whole sequence × T, shared block
- **C** `slot_recurrent_5m` — FAR² topology (encode once → slot workspace →
  cross-attend H → decode), **zero FAR² operators**. Param-identical to
  far2_5m; `slot_recurrent_bind` is the same control on the competitive
  workspace.
- **D** `far2_full` / `far2_bind` — C + adaptive representation +
  reconsideration + epistemics + protected goal (+ competitive binding
  substrate for the `_bind` variants)

**D − C is the FAR² contribution.** Without C, any win could be credited to the
recurrent latent bottleneck alone.

## The campaign funnel

Training is expensive and selective; evaluation is cheap and large. Runs earn
promotion (`scripts/run_campaign.py`):

| Stage | Budget | Purpose |
|---|---|---|
| Smoke | 200–500 steps (`smoke_test.py`, `run_campaign.py --mini`) | catch code/training failures — every script end-to-end |
| Pilot | ~2.5–3k steps, seed 0, evals 300–500 | does the mechanism show ANY signal? |
| Confirm | 5–10k steps, evals in the thousands | establish the result — survivors only |
| Seeds | 3–5 seeds | statistical confidence — important survivors only |
| Grid | the ~17-arm ablation matrix | causal proof — only after core gates pass |

The pilot runs the killer-result chain in order, with stop rules:
P0 baselines → P1 recurrence [STOP] → P2 depth extrapolation → P3a oracle
quartet on the binding substrate [STOP if the benchmark does not require
representation] → R0/R1/R4 probes → P3b autonomous controller quartet →
R8 usage gates + R9 do() interventions → cross-domain transfer.
Deferred until the core survives: P5a/P5b, P6, P7, the full grid, seeds.

## The R-sequence (representation substrate program)

| R | What it establishes | Where |
|---|---|---|
| R0 | the probe pipeline recovers PLANTED encodings (>95%) and scores chance on shuffled labels — the instrument works | `run_probes.py` (runs first, refuses to report otherwise) |
| R1 | WHERE pair values actually live: bound slot / wrong slot / pooled Z / encoder H (+ nonlinear) → Case A distributed / B bottleneck / C upstream / D nonlinear | `run_probes.py` |
| R2 | whether extra representation is even NECESSARY: fixed4 / fixed16 / oracle-EXPAND / oracle-SPLIT — if fixed4 ≈ fixed16 the benchmark is the problem; **stop** | `run_phase3a.py` |
| R3/R8 | natural operator demand BEFORE causal claims: UsageRate ≥ 15%, FalseUsage ≤ 5% — else USAGE-FAILURE, a finding, not a missing test | `run_interventions.py` |
| R4 | strict binding gates: bound ≥ chance+20pts, bound−wrong ≥ 15pts (McNemar) | `run_probes.py --strict` |
| R5 | the competitive-binding substrate itself: evidence tokens choose their owner slot; 2 register slots absorb background; binding matrix B[e,s] drives collision detection and directed SPLIT | `far2_bind` presets |
| R6 | binding curriculum (annealed 0.5 → 0.25 → 0; zero before transfer claims) | curriculum stages |
| R7 | the FREE controller in the necessity-vs-allocation quartet | `run_phase3b.py` |
| R9 | episode-level do(): block/force operators in exact replays | `run_interventions.py` |

## Phase table

| Phase | Script | Experiment | Gate (pre-registered defaults) |
|---|---|---|---|
| P0 | `run_phase0.py --model {transformer,recurrent,slot_recurrent}` | baselines A, B, C on M0–M9 | each ≥95% easy bucket; params 4.2–5.8M; FLOPs logged |
| P1 | `run_phase1.py` | stable soft-state recurrence, T=4 vs T=1 control, exact-residual telemetry | no divergence; state RMS bounded; T4 ≥ T1+5pts hard bucket; no collapse at T∈{6,8}. **Stop rule.** |
| P2 | `run_phase2.py` | trained ≤ frontier (carry ≤6, expr ≤6, mul ≤3), eval BEYOND: carry {2,6,12,24}, expr {4,8,12,16}, mul {2,3,4,6}, deriv {4,8,12} at T ∈ {2,4,8,16,32} + the 32-digit L-matched counterexample (D=0 vs D=24) | OOD 2<4<8 (≥+3pts); 16 holds; length alone must not be the cost. 8>16>32 failure → contingency ladder |
| P3a | `run_phase3a.py [--substrate bind]` | R2 oracle-necessity quartet on symbreak+linsys | fixed16 > fixed4 +5pts (**stop rule**: else the benchmark doesn't require representation); oracle-split ≥ fixed4+5pts and ≈ fixed16; binding diagnostics recorded |
| P4 | `run_phase4.py` | signal tournament (before any free controller) | best AUROC > 0.55; writes `runs/selected_signal.json` |
| P3b | `run_phase3b.py [--substrate bind]` | **free** controller in the quartet: fixed4 / rate-matched **random** / fixed16 / adaptive, + surface regression slots~(R,L) | adaptive ≥ fixed16 at ≤70% slot-compute; adaptive > random +3pts on distinction queries; split P≥0.6/R≥0.5; slots track R not L; binding separation > control rate; quartet reading (R, ρ) printed |
| P5a | `run_phase5a.py` | RECONSIDER on extraneous roots at fixed T=8: control / rate-matched random-reset / learned, + control@T12 | recovery ≥ control+5pts; ≥ random-reset+3pts; @T8 ≥ control@T12−1pt; fires 0.05–6/ep; accuracy held |
| P5b | `run_phase5b.py` | learned HALT vs free convergence-rule control | mean iters < budget; within 2pts of the r_t<0.1 rule — if the free rule wins, we say so |
| P6 | `run_phase6.py` | epistemic state on linsys: DERIVED / UNDERDETERMINED / NO_SOLUTION | unsupported ≤10% AND over-abstention ≤10%; answerable acc −≤2pts |
| P7 | `run_phase7.py` | protected goal on constrained optimization (infeasible-claim trap) | violations ≤ 0.5× control AND contrast p<0.05; accuracy held |
| Causal | `run_interventions.py` | R8 usage gates → R9 do(): block/force SPLIT/EXPAND (symbreak), RECONSIDER (extraneous) in exact replays | UsageRate ≥15% & FalseUsage ≤5% (else USAGE-FAILURE); blocking flips ≥10%; placebo ≤15%; four-state verdict per op |
| Probes | `run_probes.py [--strict]` | R0 selfcheck → R1 location probes → Case A–D diagnosis → R4 strict binding gates | R0 must pass to report anything; baseline +10pts / strict +20pts & separation 15pts |
| Final | `run_ablations.py` (matrix × seeds incl. bind arms; `--tier diagnostic` to triage; step- or `--flop-budget`-matched) then `run_transfer.py` | ablations + controls + **held-out cross-family transfer** | see below |

## The headline experiment: `run_transfer.py`

Four held-out mathematical families embed trained cognitive conditions inside
unseen surface grammars:

- `sym_poly` — symmetric POLYNOMIAL constraints broken late (SPLIT condition)
- `rational_trap` — excluded-value trap `(x²−a²)/(x−a)`, tempting root
  invalid (RECONSIDER condition)
- `iter_apply` — k-fold function application (recurrence-depth condition)
- `congruence` — modular systems that are provably underdetermined or
  inconsistent (epistemic condition)

The family token is shuffled everywhere, so nothing announces the condition.
The result that matters most: **a ~5M model detects that two mathematical
variables require independent representation, restructures its latent
workspace, solves a problem the matched slot-recurrent model fails, and
transfers that operation to an unseen mathematical structure** — splits track
`requires_distinction` by ≥10pts with binding separation, the excluded root
is discarded, modular ambiguity is abstained on — where slot-recurrent C
cannot. That, with seed error bars, is the first evidence of a cognitive
mechanism rather than benchmark exploitation.

## The controls matrix (v0.3 machinery, v0.4 families)

Standard: **if FAR² wins, there should be almost no easy alternative
explanation for why it won.** Each control removes one:

| Alternative explanation | Control that removes it |
|---|---|
| Latent-bottleneck topology, not FAR² operators | `slot_recurrent_5m` / `slot_recurrent_bind` (baseline C, param-matched) |
| Better numeric tokenization, not cognition | shared digit-sequence vocab + Abacus significance embedding in ALL four models |
| More capacity, not adaptive representation | fixed4 / fixed16 endpoints + ≤70% slot-compute gate |
| Extra slots/noise injection, not decisions | `far2_random_expand` / `far2_bind_random_expand` — rate-matched random policy, random split source |
| Adaptation couldn't help even with perfect decisions | `far2_oracle_expand` / `far2_bind_oracle` — condition-forced upper bound (never a "win") |
| Longer context = harder, so length explains depth | D ⊥ L by construction: exact carry chains at fixed digit count; multi-digit-leaf expression trees; the 32-digit L-matched counterexample |
| Generic perturbation, not intelligent reconsideration | `far2_random_reset` — rate-matched random partial reset |
| Thinking longer, not changing direction | P5a control also evaluated at T=12 |
| Task-ID shortcut, not condition detection | family token shuffled everywhere + held-out transfer families |
| Learned halting beaten by a cheap rule | P5b convergence-rule control (r_t < 0.1), same weights |
| Slot-count increase, not concept separation | binding matrix diagnostics + R0–R4 probe ladder + do() interventions |
| Answer-slot count leaks magnitude | zero-padded fixed-width answers per surface |
| Extra clean channel, not goal protection | `far2_goal_scrambled` — channel intact, wrong-episode goal |
| Aux-loss regularization, not epistemic mechanism | `far2_epi_shuffled` — labels permuted, head kept |
| More optimization compute | `--flop-budget` training-FLOP-matched runs |
| Adaptive efficiency asserted, not measured | per-episode inference FLOPs → accuracy/FLOP curves |
| Model-level ablation ≠ decision-level causality | `run_interventions.py` — R8 usage gates, then block/force in exact replays |
| Operator "used" claimed without usage | R8: UsageRate/FalseUsage gates with USAGE-FAILURE verdict |

## Anti-shortcut measures (always on)

- Family token shuffled per episode (train + eval); `family_token_mode="true"`
  exists only as the leak ablation.
- `requires_distinction` / `reconsider_label` are **task conditions used for
  curriculum and evaluation** — never claimed as internal-state oracles.
  Internal claims go through the binding matrix (`binding` record field).
- Answers are zero-padded digit sequences: slot count never encodes magnitude.
- Per-iteration exact residuals are TELEMETRY ONLY — the model never sees them.
- Both step-matched and training-FLOP-matched comparisons are supported
  (`--flop-budget`); episode records carry inference FLOPs.

## 5M acceptance gate (v0.4)

**Three mandatory claims** (confirm stage, rules in `docs/CRITERIA.md`):
1. computation adaptation — iterations buy OOD mathematical depth, and the
   L-matched counterexample shows length alone is not the cost;
2. representation adaptation — adaptive beats rate-matched random and tracks
   R (not L) at ≤70% of fixed16 slot-compute;
3. cross-domain operator transfer — unseen-family splits track the condition
   and trap recovery beats slot-recurrent C.

**Plus ≥2 of**: reconsideration beats matched compute · epistemics cut
unsupported assertions without over-abstention · goal protection halves
violations · learned halting survives its rule control · full do() causal
chain (R8 PASS + necessity + placebo) · R4 probe separation on the binding
substrate. Every claimed mechanism must beat `slot_recurrent`-anchored
controls with its random-policy arm losing; the transfer result is reported
either way, with seed error bars.

## Out of scope at 5M

MoE · SSM · world model · retrieval · background cognition · persistent/plastic
memory · skill compilation · approval systems · chat fine-tuning · scaling to
30M before the substrate program (R0–R9) produces a surviving core result.
