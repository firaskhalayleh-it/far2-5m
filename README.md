# FAR²-5M

**Prove the cognition mechanism works before proving language capability — on
exact mathematics.**

A ~5M-parameter testbed for the FAR² cognitive architecture: adaptive recurrence over a slot state,
dynamic representation (KEEP/EXPAND/SPLIT) on an explicit-binding workspace, reconsideration,
epistemic state, and a protected goal — each mechanism built behind a config flag, measured against
matched baselines, and killed if it loses its causal ablation. All tasks are exact mathematics
(v0.4): numbers are digit sequences with a shared Abacus significance embedding in every model,
every family has an independent exact solver, and every gate is a pre-registered statistical rule.
The v0.4.52 capability contract additionally separates fresh-ID competence,
OOD extrapolation, recurrent computation after Z0, and frozen zero-update
transfer; smoke accuracy cannot promote the architecture.

- Research log (dated, append-only — the current record): [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)
  · how experiments are run: [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md)
  · phases & gates: [docs/ROADMAP.md](docs/ROADMAP.md)
  · decision rules (the math): [docs/CRITERIA.md](docs/CRITERIA.md)
  · decisions log: [docs/DESIGN.md](docs/DESIGN.md)

## Research status (2026-10-18)

**The question:** can a ~5M-parameter recurrent slot model keep several
interacting variables apart and compute over them — on a benchmark built so
that no shortcut works — where a matched Transformer cannot?

**Where it stands:** the substrate now partly learns the flagship task, but the
decisive comparisons have not been run. Nothing below is a claim yet: every
number is a **single-seed, 1,000-update screen** (label-free, batch 128, T=8,
raw weights; fresh-ID symbreak R∈{1,2,4}, 768 episodes / 1,749 broken
answers). Claims need the seed counts in [docs/CRITERIA.md](docs/CRITERIA.md).

### What the screens support

- **A measurement bug came first.** Early checkpoints were scored on
  EMA(0.999) weights that were mostly initialization: 30.4% vs 53.1% raw
  token accuracy at 1,000 updates.
- **Capped ownership fixed slot collapse.** Eight variable pairs had been
  crammed into 4.3 slots, and 97% of wrong answers were another variable's
  value. With capped ownership: 86.0% token, 39.8% exact, 7.9 slots used.
- **The encoder needs local mixing.** In an isolated encoder bench
  (`diagnostics/encoder_bench.py`), every one-layer variant without local
  mixing stays near chance (≤55%) on the cross-row relation the slot needs;
  local mix + QK-norm solves it by step 100, and a plain two-layer encoder only
  by step 600.
- **Entity-steered routing is the best arm so far:**

  | Arm | Exact | Broken pairs | Roots found | Root selection |
  |---|---:|---:|---:|---:|
  | capped ownership | 41.1 | 29.7 | 59.0 | 50.3 |
  | + encoder local mix + QK-norm | 41.5 | 30.1 | 55.8 | 53.9 |
  | + entity-steered routing | **55.2** | **41.3** | **79.3** | 52.1 |

- **Recurrence is doing work on that arm:** exact accuracy rises from 18.9% at
  T=0 to 55.2% at T=8 (it collapses at T=32).

### What failed, and the open blocker

- **SPLIT is frozen.** Perfect-timing *oracle* SPLIT scores 11.3% exact against
  55.2% without it, on both substrates. The adaptive-representation operator
  is not part of current experiments.
- **Root selection is stuck at chance** (50–54% on every arm): the model finds
  the roots of a pair but not which root belongs to which variable. Diagnosed
  as a readout chicken-and-egg — the decoder would need an XOR of query
  identity and direction inside a shared slot.
- **Binding memory** (branch `exp/binding-memory`, not merged) gives the first
  selection above chance, **71.6%**, but breaks binding: 3.1% exact, 28.2%
  broken pairs. Its structural-role keys are nearly collinear (mean |cos|
  0.73), so the write leaks across pairs. Next, pre-registered: keys from
  variable identity only, with the structural role moved into the stored value.

### Not yet shown

- **No win over a tuned, compute-matched Transformer.** The only same-data
  comparison (Transformer 26.1% token, 0% exact) used an untuned baseline and
  is weak evidence.
- Autonomous allocation, causal interventions, and held-out transfer have not
  been run; the campaign pilot (`run_campaign.py`) has not been launched.

Log entries before 2026-09-25 ran from an uncommitted tree; their code is the
flag set committed with them. From 2026-09-25 on, every run writes a manifest
pinning commit, command, environment and seed.

### Branches

| Branch | Role |
|---|---|
| `far2` (default) | integration branch of the 5M research line; substrate repairs merged in, tag `r/2026-09-25-substrate-routing` |
| `exp/binding-memory` | binding-memory line, stacked on the repairs; open, not merged |
| `exp/substrate-repairs` | merged into `far2` |

Each experiment is pre-registered on an `exp/*` branch, built behind a config
flag, committed before it runs, measured with `diagnostics/measure.sh` (which
refuses a dirty tree), recorded in the research log, tagged if it is a
milestone, and returned to `far2` by pull request — failures included. Details:
[docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md).

## Setup

```bash
pip install -e ".[dev]"     # torch >= 2.2 (MPS-capable on Apple Silicon), matplotlib, pytest
```

## Follow the steps

```bash
# 0. Everything real? (14 generators audited by independent exact solvers,
#    gradients flow, training learns, stats pinned to hand-computed values)
python -m pytest

# 1. Isolated representation laws MUST pass before training. smoke_test runs
#    this preflight itself, but it is also callable independently.
python scripts/representation_preflight.py

# 2. Minutes-long end-to-end proof (preflight first, then all 5 model kinds
#    preserve core exact-task competence)
python scripts/smoke_test.py

# 3. Eyeball the data (M0-M9 + the four held-out transfer families)
python -m far2.data.registry --show-samples

# 4. First REAL-task capability gate: staged, answer-only fixed14; paired
#    Z0/T1/T4/T8; learned-seed/all-cycle diagnostics; matched Transformer and
#    parameter-identical slot-recurrent control. This is not a smoke test.
python scripts/run_foundation_curve.py --steps 4000 --eval-samples 400

# 5. The campaign funnel — smoke every script, then the pilot chain with stop
#    rules (P0 -> P1 -> P2 -> P3a bind quartet -> probes -> P3b -> do() -> transfer)
python scripts/run_campaign.py --mini     # isolated chain smoke in runs_mini/ (~45-55 min)
python scripts/run_campaign.py            # the real pilot, seed 0 (~7-9 h on an M3)

# ...or any phase by hand — each prints its PASS/FAIL gate table (docs/ROADMAP.md)
python scripts/run_phase0.py --model transformer      # baseline A
python scripts/run_phase0.py --model recurrent        # baseline B
python scripts/run_phase0.py --model slot_recurrent   # baseline C = FAR² topology, no operators
python scripts/run_phase1.py                          # stable recurrence (stop rule)
python scripts/run_phase2.py                          # depth extrapolation + L-matched counterexample
python scripts/run_phase3a.py --substrate bind        # R2 oracle-necessity quartet (stop rule)
python scripts/run_phase4.py                          # signal tournament -> selected_signal.json
python scripts/run_phase3b.py --substrate bind        # R7 FREE controller in the quartet
python scripts/run_phase5a.py                         # RECONSIDER on extraneous roots (deferred tier)
python scripts/run_phase5b.py                         # learned halting (deferred tier)
python scripts/run_phase6.py                          # epistemic state on linear systems (deferred tier)
python scripts/run_phase7.py                          # protected goal on optimization (deferred tier)

# 6. Causal + representation evidence on the trained checkpoints:
#    R8 usage gates -> R9 episode-level do(); R0 selfcheck -> R1 location probes -> R4 strict
python scripts/run_interventions.py --run runs/phase3b_adaptive_bind_seed0
python scripts/run_probes.py --strict --run runs/phase3b_adaptive_bind_seed0 \
    --control runs/phase3a_fixed14_bind_seed0

# 7. The ablation & controls matrix (--tier diagnostic to triage, --flop-budget
#    for FLOP-matched), then the HEADLINE experiment: held-out mathematical transfer
python scripts/run_ablations.py
python scripts/run_transfer.py
python scripts/make_report.py runs/
```

Anti-shortcut measures are always on: the family token is shuffled per episode
(train and eval), `requires_distinction`/`reconsider_label` are task conditions
— never internal-state oracles — answers are zero-padded so slot count never
leaks magnitude, per-iteration exact residuals are telemetry the model never
sees, and the contextual entity path plus binding matrix back any SPLIT claim.
Every gate is a
pre-registered **statistical decision rule** (margin AND significance: Wilson
CIs, exact binomial/McNemar, two-proportion z, surface-slope OLS —
[docs/CRITERIA.md](docs/CRITERIA.md), implemented in `far2/eval/stats.py`),
operator verdicts use PASS / FAIL / NOT-TESTABLE / USAGE-FAILURE, and every
FAR² win must survive its causal controls: rate-matched random policies, the
oracle upper bound, scrambled-goal / shuffled-epistemic arms
([docs/ROADMAP.md](docs/ROADMAP.md) controls matrix). `python -m far2.presets
--table` prints the parameter ledger the docs are generated from.

[docs/RESULTS.md](docs/RESULTS.md) is the 2026-08-29 validation snapshot from
before the substrate repairs; it keeps smoke success separate from failed or
not-testable scientific gates. Later results are in
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

Every script accepts `--steps N --eval-samples N --device cpu|mps` for scaled-down runs
(`--mini` presets exist for quick sanity passes). All runs land in `runs/<name>/` with
`config.json`, `train_log.jsonl`, `eval/*.jsonl`, and checkpoints; `make_report.py` turns any set of
runs into the research dashboard.

## Repo map

```
far2/data      shared 72-token math vocabulary + 14 generators: M0-M9 trainable
               (carry_add, multiply, base_arith, expr_eval, linsys, symbreak,
               rewrite, differentiate, extraneous, optimize) + 4 held-out
               transfer (sym_poly, rational_trap, iter_apply, congruence) —
               exact ground truth, seeded, solver-audited
far2/models    shared layers (incl. Abacus significance embedding, contextual
               entity encoder, and explicit concept binding) · transformer &
               recurrent baselines · FAR² core (slot workspace, operators,
               binding matrix, do() hooks)
far2/train     curriculum (stages A-F, operator unlocks, entity/binding anneal) ·
               trainer (losses incl. entity_nll/bind_nll, EMA, depth sampling,
               logs)
far2/eval      episode harness (JSONL telemetry incl. D/R/L, residual_traj,
               binding) · metrics & Gate · stats (the decision rules) ·
               probes (R0/R1/R4)
scripts        run_campaign.py funnel + one script per phase + ablations +
               interventions + probes + transfer + smoke + report
diagnostics    experiment scripts, benches and probes · measure.sh (refuses a
               dirty tree, writes run manifests) · PREREGISTERED_*.md · results/
               (committed run logs + manifests)
tests          pytest: the "not mockers" layer (153 tests incl. scientific contracts)
```
