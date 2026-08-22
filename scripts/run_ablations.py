"""The ablation-and-controls matrix (blueprint Part X): baselines, mechanism
knock-outs, AND the causal controls that remove easy alternative explanations
(random policies, oracle upper bound, scrambled goal, shuffled epistemic
labels), across seeds. If removing a mechanism changes nothing, it does not
survive to 30M.

DATA MATCHING: every arm — baselines included — trains on the IDENTICAL
curriculum (same stages, same mixes, same sampler seed → the same episodes);
only the operator unlocks each architecture can express differ. Without this,
D−C would measure task exposure, not FAR² operators.

HOLDOUT FREEZE: the transfer families are NOT evaluated here by default —
repeated ablation-time peeking would allow selection against the holdout.
--include-transfer unfreezes them for FINAL CONFIRMATION ONLY.

Tiers: --tier diagnostic (1 seed, 1/4 steps, half eval) triages which
comparisons deserve full runs; --tier full is the reportable matrix.

Compute matching: default is parameter/data-matched (same steps for all).
--flop-budget <flops> switches to training-FLOP-matched — each preset's steps
are set so total training compute is equal (the recurrent models process more
FLOPs per step, so they get fewer steps)."""

import json
import os
import statistics
from dataclasses import replace

from _common import apply_mini, make_tc, parser, run_eval, train

from far2.data.registry import GENERATORS, TRANSFER_GENERATORS
from far2.eval import EvalSpec
from far2.eval.metrics import (accuracy, acc_per_flop, assertion_rates,
                               binding_separation, goal_violation_rate, mean,
                               recovery_rate, split_selectivity)
from far2.models import build_model
from far2.presets import get_preset
from far2.train import FULL_CURRICULUM, Curriculum

ABLATION_PRESETS = [
    # baselines
    "transformer_5m", "recurrent_5m", "slot_recurrent_5m",
    # the model + capacity quartet endpoints
    "far2_full", "far2_no_adaptive", "far2_fixed4",
    # causal controls: decisions vs noise vs perfect decisions
    "far2_random_expand", "far2_oracle_expand", "far2_no_split",
    # reconsideration
    "far2_no_reconsider", "far2_random_reset", "far2_fixed_recur",
    # epistemics: mechanism vs aux-loss regularization
    "far2_no_epistemic", "far2_epi_shuffled",
    # goal: protection vs extra clean channel
    "far2_no_goal", "far2_goal_scrambled",
    # binding substrate (R5+): competitive binding vs its own controls
    "far2_bind", "far2_bind_random_expand", "far2_bind_oracle",
    "slot_recurrent_bind",
]
# arms that are a preset + a TrainConfig override rather than a ModelConfig
SPECIAL_ARMS = {"far2_epi_shuffled": ("far2_full", {"epistemic_shuffled": True})}
# arms that consult ground truth at run time: upper bounds, never "wins"
ORACLE_ARMS = {"far2_oracle_expand", "far2_bind_oracle"}
# random-policy arms are only a control when their firing rates equal the
# MEASURED rates of the learned arm they refute — preset defaults are just
# placeholders. Each maps to the same-seed learned reference it calibrates
# against; the reference must run FIRST in the same invocation.
RANDOM_ARM_REFERENCE = {
    "far2_random_expand": "far2_full",
    "far2_random_reset": "far2_full",
    "far2_bind_random_expand": "far2_bind",
}
TYPICAL_LEN = 120


def resolve_arm(name: str):
    preset, tc_over = SPECIAL_ARMS.get(name, (name, {}))
    return get_preset(preset), tc_over


def capabilities(mc) -> set[str]:
    """Operator unlocks this architecture can express — the ONLY thing that
    may differ between arms' curricula."""
    caps: set[str] = set()
    if mc.kind != "far2":
        return caps
    disabled = set(getattr(mc, "disable_actions", ()) or ())
    if mc.adaptive_slots:
        caps |= {"expand", "split"} - disabled
    if mc.halting == "learned":
        caps.add("halt")
    if mc.reconsider:
        caps.add("reconsider")
    if mc.epistemic:
        caps.add("epistemic")
    return caps - disabled


def matched_stages(mc):
    """The one shared curriculum: identical stages/mixes/knobs for every arm;
    unlocks intersected with the arm's capabilities (ipt has no meaning for
    the non-recurrent transformer)."""
    caps = capabilities(mc)
    out = []
    for st in FULL_CURRICULUM:
        over = {"unlocked": set(st.unlocked or set()) & caps}
        if mc.kind == "transformer":
            over["ipt_alpha"] = 0.0
        out.append(replace(st, **over))
    return out


def eval_grid(n: int, include_transfer: bool = False) -> list[EvalSpec]:
    fams = dict(GENERATORS)
    if include_transfer:                 # FINAL CONFIRMATION ONLY
        fams.update(TRANSFER_GENERATORS)
    out = []
    for f, g in fams.items():
        mid = (g.min_difficulty + g.max_difficulty) // 2
        out.append(EvalSpec(f, sorted({g.min_difficulty + 1, mid,
                                       g.max_difficulty}), n_samples=n))
    return out


def flop_matched_steps(mc, batch: int, budget: float) -> int:
    model = build_model(mc)
    per_step = 3 * batch * model.flops_per_sample(TYPICAL_LEN, mc.n_iters_train)
    return max(200, int(budget / per_step))


def summarize(name: str, seed: int, recs: list[dict], trainer) -> dict:
    trainable = [r for r in recs if r["family"] in GENERATORS]
    transfer = [r for r in recs if r["family"] in TRANSFER_GENERATORS]
    unsupported, over = assertion_rates(trainable)
    sel = split_selectivity(trainable)
    return {
        "preset": name, "seed": seed,
        "acc_overall": accuracy(trainable),
        "acc_transfer": accuracy(transfer),
        "acc_by_family": {f: accuracy(recs, family=f)
                          for f in list(GENERATORS) + list(TRANSFER_GENERATORS)},
        "acc_per_flop": acc_per_flop(trainable),
        "split_precision": sel["precision"], "split_recall": sel["recall"],
        "transfer_split_selectivity": split_selectivity(transfer),
        "binding": binding_separation(trainable),
        "recovery_rate": recovery_rate(trainable),
        "transfer_recovery": recovery_rate(transfer),
        "unsupported_assertion": unsupported, "over_abstention": over,
        "goal_violation": goal_violation_rate(trainable),
        "params": trainer.model.param_count(),
        "train_flops": trainer.train_flops,
    }


def main():
    p = parser("Ablation & controls matrix x seeds on the full curriculum",
               steps=6000, eval_samples=150)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--presets", nargs="+", default=ABLATION_PRESETS)
    p.add_argument("--flop-budget", type=float, default=None,
                   help="total training FLOPs per run (switches to FLOP-matched)")
    p.add_argument("--tier", choices=["full", "diagnostic"], default="full",
                   help="diagnostic: 1 seed, 1/4 steps, half eval — triage "
                        "which comparisons to promote to full runs")
    p.add_argument("--include-transfer", action="store_true",
                   help="UNFREEZE the held-out transfer families — final "
                        "confirmation only; never during development/triage")
    args = p.parse_args()
    apply_mini(args)
    if args.tier == "diagnostic":
        args.seeds = args.seeds[:1]
        args.steps = max(800, args.steps // 4)
        args.eval_samples = max(60, args.eval_samples // 2)
        print(f"[diagnostic tier] seeds={args.seeds} steps={args.steps} "
              f"eval_samples={args.eval_samples}")
    if args.mini:
        args.seeds = args.seeds[:1]

    summaries = []
    measured: dict[tuple[str, int], dict[str, float]] = {}
    for preset in args.presets:
        for seed in args.seeds:
            args.seed = seed
            mc, tc_over = resolve_arm(preset)
            ref = RANDOM_ARM_REFERENCE.get(preset)
            if ref is not None:
                if (ref, seed) not in measured:
                    raise SystemExit(
                        f"[ablations] {preset} calibrates its random rates to "
                        f"the MEASURED usage of '{ref}' seed{seed}, which has "
                        "not been trained in this invocation — put the "
                        "reference arm before its random control in --presets.")
                r = measured[(ref, seed)]
                over = {}
                if getattr(mc, "repr_policy", None) == "random":
                    over.update(random_expand_p=r["expand"],
                                random_split_p=r["split"])
                if getattr(mc, "reconsider_policy", None) == "random":
                    over["random_reconsider_p"] = r["reconsider"]
                mc = replace(mc, **over)
                print(f"  [{preset}] rates calibrated to {ref} seed{seed} "
                      "(raw, no floor): "
                      + " ".join(f"{k}={v:.4f}" for k, v in over.items()))
            steps = (flop_matched_steps(mc, args.batch, args.flop_budget)
                     if args.flop_budget else args.steps)
            saved = args.steps
            args.steps = steps
            tc = make_tc(args, f"ablate_{preset}_seed{seed}", ipt_alpha=0.5,
                         **tc_over)
            args.steps = saved
            # IDENTICAL curriculum for every arm — only unlocks differ
            stages = Curriculum(matched_stages(mc), tc.steps)
            trainer = train(mc, tc, stages)
            recs = run_eval(trainer,
                            eval_grid(args.eval_samples, args.include_transfer),
                            "ablation")
            if preset in set(RANDOM_ARM_REFERENCE.values()):
                iters = max(mean(recs, "iters_used"), 1.0)
                measured[(preset, seed)] = {
                    "expand": min(mean(recs, "expand_count") / iters, 1.0),
                    "split": min(mean(recs, "split_count") / iters, 1.0),
                    "reconsider": min(mean(recs, "reconsider_count") / iters,
                                      1.0)}
            s = summarize(preset, seed, recs, trainer)
            s["oracle"] = preset in ORACLE_ARMS
            summaries.append(s)
            print(f"  {preset} seed{seed}: acc {s['acc_overall']:.1%} "
                  f"transfer {s['acc_transfer']:.1%} "
                  f"({steps} steps, {s['train_flops']:.2e} train FLOPs)")

    out_path = os.path.join(args.out, "ablation_summary.json")
    with open(out_path, "w") as fh:
        json.dump(summaries, fh, indent=2)

    mode = "FLOP-matched" if args.flop_budget else "step-matched"
    if args.tier == "diagnostic":
        mode += " · DIAGNOSTIC tier (promote interesting comparisons to full)"
    print(f"\n[{mode}] {'preset':<21} {'acc':>8} {'±':>6} {'transfer':>9} {'acc/PFLOP':>10}")
    for preset in args.presets:
        rows = [s for s in summaries if s["preset"] == preset]
        if not rows:
            continue
        accs = [s["acc_overall"] for s in rows]
        trans = [s["acc_transfer"] for s in rows if s["acc_transfer"] == s["acc_transfer"]]
        apf = [s["acc_per_flop"] * 1e15 for s in rows]
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        tm = statistics.mean(trans) if trans else float("nan")
        tag = preset + ("†" if preset in ORACLE_ARMS else "")
        print(f"{tag:<21} {statistics.mean(accs):>8.1%} {sd:>6.1%} "
              f"{tm:>9.1%} {statistics.mean(apf):>10.2f}")
    if any(s["oracle"] for s in summaries):
        print("† consults ground truth at run time — an upper bound, never a win")
    print(f"\nSummary written to {out_path}. Key comparisons: far2_full vs "
          "slot_recurrent_5m (D−C) · far2_full vs far2_random_expand "
          "(decisions vs noise) · far2_oracle_expand (headroom) · "
          "far2_no_epistemic vs far2_epi_shuffled (mechanism vs aux loss) · "
          "far2_no_goal vs far2_goal_scrambled (protection vs channel). "
          "Then: run_interventions.py + run_probes.py for causal/representation "
          "evidence; run_transfer.py for the headline unseen-family result; "
          "make_report.py for the dashboard.")


if __name__ == "__main__":
    main()
