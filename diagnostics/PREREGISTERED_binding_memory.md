# Pre-registration: binding memory (written 2026-09-25, before any run)

## Hypothesis
Root selection sits at chance because, with both variables of a pair in one slot,
choosing the root is an XOR of the queried variable's identity and the hint's
direction; an XOR gives first-order gradients that average to zero (measured
hint-row pull ~1e-7 vs 1.5e-5 for the product row). A role-keyed binding memory
whose roles are *structural* (variable id + one feature per (clause operator,
operand position) occurrence, from token ids only) makes selection linear:
"I am operand 1 of the GEQ row -> take the larger root". With *identity* roles
the memory is equivalent to today's decoder and should not help.

## Arms (1,000 updates, label-free, single-stage P3 mix, seed 0, raw weights, T=8)
Base = the `route` arm (capped ownership, enc_local_mix + enc_qk_norm,
slot_query_entities). Reference numbers, route seed 0: exact 55.2, symmetric 98.6,
broken 41.3, selection 52.1, half-sum 8.1, other-pair 8.6, T0->T8 exact +36.3.
- `bm_identity`: bind_memory, identity roles.
- `bm_structural`: bind_memory, structural roles.

## Predictions that can fail
| # | Prediction | bm_structural | bm_identity |
|---|---|---|---|
| P1 | selection correct/(correct+partner) | >= 65% | < 60% |
| P2 | broken accuracy | >= 50% | (report) |
| P3 | exact (secondary: seed range ~10 pts) | >= 60% | (report) |
| P4 | workspace dependence: slots rolled across episodes before every memory write -> broken <= 15% and symmetric <= 20% | required | required |
| P5 | recurrence still works: exact(T8) - exact(T0) >= 15 pts | required | (report) |

If bm_identity also reaches P1, the XOR diagnosis is wrong and is reported as such.

## Guards
- symmetric >= 97.0% (route 98.6%)
- other-pair share <= 12% and half-sum share <= 12%
- roles are built from token ids only; a test asserts identical roles for two
  episodes that differ only in digits.

## Scope
Substrate change shared with slot_recurrent_bind: a pass credits no FAR² operator.
Structural roles are an input-syntax featurization (like digit significance) and
must be offered to the transformer baselines before any cross-model claim.
One seed is a screen.
