# Research log

Dated, append-only. Each entry: what ran, numbers, verdict, what it rules out.
Contract unless stated: label-free, 1,000 updates, single-stage P3 mixture,
batch 128, T=8, raw weights, fresh-ID symbreak R∈{1,2,4} (768 episodes, 1,749
broken answers). Logs: `diagnostics/results/`. Entries before 2026-09-25 (routing)
ran from an uncommitted tree; their code is the flag set committed with them.

## 2026-09-20 — EMA artifact and ownership collapse
- Early checkpoints are scored on EMA(0.999, no bias correction) weights that are
  mostly initialization: 30.4% vs 53.1% raw token accuracy at 1,000 updates.
- 97% of wrong answers were another variable's value; 8 pairs used 4.3 of 12 slots.
- **Capped ownership** (`bind_norm="capped"`, 32 sweeps): token 86.0%, exact 39.8%,
  7.9 slots used. Matched transformer on the same data: 26.1% token, 0% exact
  (untuned; weak evidence until a tuned, compute-matched baseline runs).

## 2026-09-25 — selection failure, encoder and routing repairs, SPLIT frozen
Baseline capped seeds 0/1: exact 41.1/31.3, broken 29.7/27.1, selection 50.3/52.4.

| Arm | Exact | Sym | Broken | Roots found | Selection | Half-sum | Other-pair | Slot order probe | Hint probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| capped (s0) | 41.1 | 95.5 | 29.7 | 59.0 | 50.3 | 18.6 | 15.8 | 48.6 | 63.5 |
| + encoder local mix + QK-norm | 41.5 | 95.8 | 30.1 | 55.8 | 53.9 | 15.4 | 21.3 | 51.2 | **100** |
| + entity-steered routing | **55.2** | **98.6** | **41.3** | **79.3** | 52.1 | 8.1 | 8.6 | 50.9 | 100 |
| + clause read | 52.9 | 98.3 | 41.7 | 81.6 | 51.1 | 6.7 | 7.9 | 50.8 | 100 |
| routing + oracle SPLIT | 11.3 | 85.5 | 20.7 | 40.9 | 50.6 | 16.6 | 32.0 | 56.3 | 100 |

- Encoder debug: the 62% hint probe was already 62% at the input embedding; encoder
  attention was uniform (neighbour ~1/178); position embeddings untrained N(0,1).
  Isolated bench: only local mixing solves the two-hop relation (cross-row 100%).
- Routing: slot attention entropy 0.99 → 0.45; own sum/product rows 17–20×, but own
  **hint row 0.94×** at every head, iteration and checkpoint; loss pull toward it
  ~1e-7 vs 1.5e-5 toward the product row. Clause read did not change this.
- Verdict: encoder check PASS; selection remains at chance. The blocker is a
  readout chicken-and-egg (the decoder would need an XOR of query identity and
  direction inside a shared slot).
- **SPLIT frozen**: perfect-timing oracle SPLIT is worse on both substrates.
- Recurrence: routing arm exact 18.9 → 55.2 from T=0 to T=8; collapses at T=32.

Next: binding memory (per-variable write/read), pre-registered in
`diagnostics/PREREGISTERED_binding_memory.md`.

## 2026-09-25 — binding memory, first run: write-rule bug (not evidence)
`bm_structural` seed 0 at commit c188495: exact 0.0%, symmetric 26.3%, broken 10.9%,
other-pair 60.4%; loss matched the route arm to step 200, then rose (1.70 → 2.04) and
ended at 1.58 vs 0.83. Debug (`diagnostics/debug_binding_memory.py`): memory norm grew
~7x per write (52 → 3.8e8 over 9 writes); keys had mean |cos| 0.5, so the batched delta
rule's step 0.88 × λmax(KKᵀ) ≈ 8.5 was far past the stability limit of 2; reads were
equally similar to every variable's filler and fillers collapsed (cos 0.99). Fix: exact
projection write M ← M + η Kᵀ(KKᵀ+εI)⁻¹(F − KM), contractive for η ∈ (0, 2]. The run is
kept as `diagnostics/results/bm_structural_seed0_delta-rule-diverged.*`; the
pre-registration is unchanged and applies to the rerun.

## 2026-09-25 — binding memory rerun (commit 6c8ddfd): selection above chance, binding broken
`bm_structural` seed 0 vs route seed 0, against `diagnostics/PREREGISTERED_binding_memory.md`:
selection **71.6%** (P1 ≥65 PASS; first arm above chance, all earlier 50–54), workspace
shuffle → broken 9.2%, symmetric 10.1% (P4 PASS), broken 28.2% (P2 FAIL), exact 3.1%
(P3 FAIL), T0→T8 +1.8 (P5 FAIL), symmetric 76.2% (guard FAIL), other-pair 38.0% (guard
FAIL), roots found 39.5% (route 79.3). Memory healthy: read vs own filler cos 1.000
(0.46 to others), norm flat ~260. Keys built from structural roles are near-collinear
(mean |cos| 0.73, max 0.96), making the exact write ill-conditioned; likely cause of the
cross-pair leakage. Verdict: the structural role fixes selection, this memory design
breaks binding. Next (to pre-register): keys from identity only with the structural
role in the filler; and an ablation giving the structural role to the route decoder
query without any memory.

## 2026-10-18 — commit IDs re-mapped (metadata-only history rewrite)
History was rewritten without changing any file: commit-message trailers were
removed, then author/committer e-mail was normalized for the public repository.
Every tree is byte-identical, so the code each run used is unchanged, but the
commit IDs recorded in the entries above and in
`diagnostics/results/*.manifest.json` no longer resolve. Map:

| Recorded | Now | Tree (unchanged) | Commit |
|---|---|---|---|
| c188495 | 3ab7e0f | db1377f7db4f | Binding memory behind a flag, pre-registered before any run |
| 6c8ddfd | 1089aa7 | 0383ae9e6dfc | Binding memory: exact projection write replaces diverging delta rule |

Check a row with `git rev-parse <Now>^{tree}`, which must match the tree column.
