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

- Full blueprint: the "FAR²-5M Blueprint" artifact · phases & gates: [docs/ROADMAP.md](docs/ROADMAP.md)
  · decision rules (the math): [docs/CRITERIA.md](docs/CRITERIA.md)
  · decisions log: [docs/DESIGN.md](docs/DESIGN.md)

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

See [docs/RESULTS.md](docs/RESULTS.md) for the dated local validation snapshot;
it keeps smoke success separate from failed or not-testable scientific gates.

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
tests          pytest: the "not mockers" layer (147 tests incl. scientific contracts)
```
