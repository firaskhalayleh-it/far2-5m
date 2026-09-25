# Pre-registered thresholds for encoder-quality fixes (written 2026-09-25, before any fix ran)

Baseline: capped ownership (bind_norm=capped, 32 sweeps), 1,000 updates, single-stage P3 mix,
raw weights, T=8, fresh-ID symbreak R in {1,2,4}, 768 episodes / 1,749 broken answers.
Two seeds only (seed 2 was stopped), so the spread is provisional.

| Metric | seed 0 | seed 1 | baseline range |
|---|---:|---:|---:|
| Exact | 41.1% | 31.3% | 31–41 |
| Symmetric | 95.5% | 92.7% | 92.7–95.5 |
| Broken | 29.7% | 27.1% | 27.1–29.7 |
| Selection correct/(correct+partner) | 50.3% | 52.4% | 50.3–52.4 |
| Half-sum share of broken | 18.6% | 20.5% | 18.6–20.5 |
| Other-pair share of broken | 15.8% | 19.6% | 15.8–19.6 |
| Probe: ordering bit in owner slot | 48.6% | 50.3% | chance |
| Probe: H at own hint-row token, "left of GEQ" | 63.5% | 63.9% | ~64 |

Binomial SE on 1,749 broken answers is about 1.1 points; the seed-to-seed range is the larger term.

## A fix counts only if, on the same contract

1. **Encoder prediction:** hint-row probe ≥ 80% (baseline ~64%).
2. **Workspace prediction:** ordering probe in the owner slot ≥ 60% (baseline chance).
3. **Selection:** correct/(correct+partner) ≥ 60% (baseline max 52.4%, ~4× the seed range above it).
4. **Broken accuracy:** ≥ 35% (baseline max 29.7%).
5. **Second prediction that can fail:** half-sum share falls below 18.6%. If selection moves but half-sum does not,
   the slot got a direction signal without separating the variables — a different finding, reported as such.

## Guards (a gain that breaks these is not a win)

- Symmetric accuracy ≥ 92.7% (baseline minimum).
- Other-pair share ≤ 19.6% (baseline maximum).
- Encoder features may only change what the encoder computes; nothing may add encoder state into the slot
  outside the existing input-warm initialization and per-iteration cross-attention.

## Honest scope

These arms change the substrate (encoder) shared by FAR² and slot_recurrent_bind; a pass credits nothing to
FAR² operators. enc2 adds 1.23M parameters and breaks parameter matching with the transformer; rope (+0) and
roles (+28K) are input-representation changes that must be offered to the baselines before any cross-model
comparison. One seed per arm is a screen, not a result.
