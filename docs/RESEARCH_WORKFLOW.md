# Research workflow

How an experiment moves from idea to recorded evidence, so every number can be
traced to the exact code, command and seed that produced it.

## Branches

| Branch | Role |
|---|---|
| `main` | stable; only merged, reviewed milestones |
| `far2` | integration branch of the 5M research line |
| `exp/<topic>` | one experiment line, branched from `far2` (or stacked on an open `exp/*` it depends on) |

An experiment returns to `far2` through a pull request whose description carries
the pre-registered checks and their verdicts, including failures.

## One experiment, in order

1. **Pre-register** on the `exp/*` branch: hypothesis, the change, predictions that
   can fail, thresholds and guards, committed as `diagnostics/PREREGISTERED_<topic>.md`
   *before* any run.
2. **Implement behind a flag** in `far2/config.py` with the default unchanged; tests
   in `tests/` prove the default path is identical and the new path behaves.
   `python -m pytest` must pass before a commit.
3. **Commit, then run.** `diagnostics/measure.sh ARM SEED` refuses to start when
   `far2/ scripts/ diagnostics/ tests/` has uncommitted changes (override:
   `FAR2_ALLOW_DIRTY=1`, which is recorded). Each run writes `run_manifest.json`
   (commit, branch, dirty state, command, `FAR2_*` env, host, duration, parsed
   metrics) beside its checkpoints.
4. **Record.** The run log and manifest are copied to `diagnostics/results/` and
   committed with a dated entry in `docs/RESEARCH_LOG.md`: what ran, the numbers,
   the verdict against the pre-registration, and what it rules out.
5. **Checkpoint.** A milestone gets an annotated tag `r/<YYYY-MM-DD>-<topic>` on the
   commit that produced it; the tag message holds the headline numbers.
6. **Pull request** into `far2`.

## What lives where

| Artifact | Location | In git? |
|---|---|---|
| Code, flags, tests | `far2/`, `tests/` | yes |
| Experiment scripts, pre-registrations | `diagnostics/` | yes |
| Run logs + manifests (small text) | `diagnostics/results/` | yes |
| Narrative results | `docs/RESEARCH_LOG.md`, `docs/RESULTS.md` | yes |
| Model weights (`*.pt`, ~46 MB each) | `runs_diag/`, `runs_*/` | no (gitignored); reproducible from commit + seed; archive a milestone's weights as a GitHub release asset if needed |

## Rules

- Never `git add -A`: the tree also holds shelved-track files (crawled text,
  labelling pages) that must not be pushed. Stage paths explicitly.
- Numbers for claims come from raw *and* EMA weights below ~3,000 updates.
- One seed is a screen; claims need the seed count in `docs/CRITERIA.md`.
- Heavy runs (> ~10 min) are stated and approved before they start.
