"""Phase 1 — stable soft-state recurrence on exact mathematics. FAR² with
everything off except the gated state loop, fixed T=4, against an identically
configured T=1 control. Also reports the exact-progress telemetry: does the
per-iteration decoded candidate get mathematically closer to correct?
Stop rule: if this gate fails, nothing else gets built. See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.data.registry import GENERATORS
from far2.eval import EvalSpec
from far2.eval.metrics import (Gate, accuracy, residual_monotonicity,
                               state_norm_stats)
from far2.presets import get_preset
from far2.train import core_mixes, single_stage
from far2.train.curriculum import TRAIN_DMAX
from far2.utils.logging import read_jsonl

# depth-graded families for the easy/hard buckets (base_arith's difficulty is
# a BASE, not a depth — it trains in the mix but is bucketed elsewhere)
DEPTH_CORE = ["carry_add", "multiply", "expr_eval", "rewrite", "differentiate"]


def specs(n):
    out = []
    for f in DEPTH_CORE:
        g = GENERATORS[f]
        hard = TRAIN_DMAX[f]
        mid = max(g.min_difficulty, (g.min_difficulty + hard) // 2)
        out.append(EvalSpec(f, sorted({g.min_difficulty, mid, hard}),
                            n_samples=n))
    return out


def bucket_acc(recs, hard: bool) -> float:
    sel = [r for r in recs
           if (r["difficulty"] >= TRAIN_DMAX[r["family"]]) == hard]
    return sum(r["correct"] for r in sel) / len(sel) if sel else float("nan")


def main():
    p = parser("Phase 1: FAR2 fixed T=4 vs T=1 control", steps=3000)
    args = p.parse_args()
    apply_mini(args)

    runs = {}
    for T in (4, 1):
        mc = replace(get_preset("far2_5m"), n_iters_train=T, init_slots=8,
                     max_slots=8)
        tc = make_tc(args, f"phase1_T{T}_seed{args.seed}")
        runs[T] = train(mc, tc, single_stage(core_mixes(1.0, hard=True)))

    recs4 = run_eval(runs[4], specs(args.eval_samples), "T4")
    recs1 = run_eval(runs[1], specs(args.eval_samples), "T1")
    # extrapolation smoke: same T=4-trained model read out at more iterations
    recs_ext = {T: run_eval(runs[4],
                            [EvalSpec(f, [TRAIN_DMAX[f]], [T], args.eval_samples)
                             for f in DEPTH_CORE], f"T4_at_{T}")
                for T in (6, 8)}

    log = read_jsonl(f"{runs[4].tc.out_dir}/train_log.jsonl")
    losses = [r["loss"] for r in log]
    mono = residual_monotonicity(recs4)
    if mono.get("n"):
        print(f"\n  exact-progress telemetry: {mono['frac_monotone']:.1%} of "
              f"episodes have non-increasing residuals; mean drop "
              f"{mono['mean_drop']:.2f} (n={mono['n']})")

    gate = Gate("Phase 1 · stable soft-state recurrence")
    gate.check("training did not diverge",
               all(l == l and l < 1e4 for l in losses),
               f"final loss {losses[-1]:.3f}")
    ns = state_norm_stats(recs4)
    gate.check("state RMS bounded", 0.5 <= ns["min"] and ns["max"] <= 8
               and abs(ns["slope"]) < 0.05,
               f"range [{ns['min']:.2f}, {ns['max']:.2f}] slope {ns['slope']:.4f} "
               f"(post_rms holds this by construction; gate bites for state_norm='none' ablations)")
    a4, a1 = accuracy(recs4), accuracy(recs1)
    h4, h1 = bucket_acc(recs4, True), bucket_acc(recs1, True)
    gate.check("T=4 matches T=1 overall", a4 >= a1 - 0.01,
               f"T4 {a4:.1%} vs T1 {a1:.1%}")
    gate.check("T=4 beats T=1 on hard bucket by >=5pts", h4 >= h1 + 0.05,
               f"hard: T4 {h4:.1%} vs T1 {h1:.1%}")
    base = bucket_acc(recs4, True)
    for T, recs in recs_ext.items():
        acc = accuracy(recs)
        gate.check(f"no collapse at T={T}", acc >= base - 0.02,
                   f"{acc:.1%} vs T4-readout {base:.1%}")
    finish(gate.report())


if __name__ == "__main__":
    main()
