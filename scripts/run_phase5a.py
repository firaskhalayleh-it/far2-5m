"""Phase 5a — RECONSIDER at a FIXED iteration budget on the extraneous-root
trap (a locally plausible candidate invalidated by the original constraint),
with the two controls that pin down what (if anything) it contributes:

    control      no reset action, T=8            (same compute)
    random_reset partial reset at RANDOM times,  (same op, no intelligence;
                 rate-matched to the learned arm  isolates decision vs
                 T=8                              perturbation)
    reconsider   learned trigger, T=8

Plus the thinking-longer check: the control is ALSO evaluated at T=12. If
plain extra iterations recover as much as reconsideration at T=8, "changing
direction" is not doing anything "thinking longer" wouldn't. Learned halting
comes after (P5b). See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec
from far2.eval.metrics import (Gate, accuracy, correct_counts, mean,
                               recovery_counts, recovery_rate)
from far2.eval.stats import non_inferiority, superiority
from far2.presets import get_preset
from far2.train import Stage, core_mixes, fam

FIXED_T = 8
LONGER_T = 12


def mix():
    return core_mixes(0.3) + [fam("linsys", 0.5), fam("extraneous", 1.5)]


def stages_for(arm: str):
    if arm == "control":
        return [Stage("fixed", 1.0, mix(), depth="fixed", ipt_alpha=0.5)]
    # random_reset trains on the IDENTICAL stages as the learned arm (same
    # data, same loss weights) — only repr_policy differs. An unmatched
    # single-stage control would let curriculum differences masquerade as
    # the learned trigger's contribution.
    return [
        Stage("warmup", 0.4, mix(), unlocked={"reconsider"}, depth="fixed",
              ipt_alpha=0.5, oracle_weight=1.0, entropy_bonus=0.01, tau=1.0),
        Stage("free", 0.6, mix(), unlocked={"reconsider"}, depth="fixed",
              ipt_alpha=0.5, oracle_weight=0.3, entropy_bonus=0.01, tau=0.7),
    ]


def eval_specs(n, T=FIXED_T):
    return [EvalSpec("extraneous", [2, 4, 6, 8], [T], n_samples=n)]


def main():
    p = parser("Phase 5a: reconsideration at fixed compute (+ random-reset "
               "and thinking-longer controls)", steps=3000)
    args = p.parse_args()
    apply_mini(args)

    base = replace(get_preset("far2_5m"), halting="fixed", init_slots=8,
                   max_slots=8, n_iters_train=FIXED_T)
    recs, trainers = {}, {}
    for arm, mc in (("reconsider", replace(base, reconsider=True)),
                    ("control", base)):
        tc = make_tc(args, f"phase5a_{arm}_seed{args.seed}", ipt_alpha=0.5)
        trainers[arm] = train(mc, tc, stages_for(arm))
        recs[arm] = run_eval(trainers[arm], eval_specs(args.eval_samples), "phase5a")

    # rate-matched random-reset control (trained AFTER the learned arm so its
    # per-iteration reset probability can be matched to measured behavior)
    # NO lower clamp: zero learned usage must be matched by zero random resets
    r_rate = min(mean(recs["reconsider"], "reconsider_count") / FIXED_T, 1.0)
    print(f"[phase5a] random-reset arm rate matched (raw, no floor): "
          f"P(reset)={r_rate:.4f}/iter")
    mc_rand = replace(base, reconsider=True, reconsider_policy="random",
                      random_reconsider_p=r_rate)
    tc_rand = make_tc(args, f"phase5a_random_reset_seed{args.seed}", ipt_alpha=0.5)
    trainers["random_reset"] = train(mc_rand, tc_rand, stages_for("random_reset"))
    recs["random_reset"] = run_eval(trainers["random_reset"],
                                    eval_specs(args.eval_samples), "phase5a")

    # thinking-longer control: SAME control weights, simply more iterations
    recs["control_T12"] = run_eval(trainers["control"],
                                   eval_specs(args.eval_samples, LONGER_T),
                                   "phase5a_T12")

    order = ["control", "random_reset", "reconsider", "control_T12"]
    rr = {a: recovery_rate(r) for a, r in recs.items()}
    rkn = {a: recovery_counts(r) for a, r in recs.items()}
    acc = {a: accuracy(r) for a, r in recs.items()}
    akn = {a: correct_counts(r) for a, r in recs.items()}
    n_rec = mean(recs["reconsider"], "reconsider_count")
    n_rand = mean(recs["random_reset"], "reconsider_count")

    print(f"\n                 accuracy   recovery   T")
    for a in order:
        T = LONGER_T if a == "control_T12" else FIXED_T
        print(f"  {a:<13} {acc[a]:>8.1%} {rr[a]:>9.1%} {T:>3}")
    print(f"  reset events/episode: learned {n_rec:.2f} · random {n_rand:.2f}")

    gate = Gate("Phase 5a · reconsideration at matched compute")
    ok, d = superiority(*rkn["reconsider"], *rkn["control"], margin=0.05)
    gate.check("recovery: reconsider > control at IDENTICAL compute (m=5pts)",
               ok, d)
    ok, d = superiority(*rkn["reconsider"], *rkn["random_reset"], margin=0.03)
    gate.check("learned trigger > rate-matched RANDOM reset (m=3pts)", ok,
               d + " — else any perturbation of equal magnitude recovers as much")
    ok, d = non_inferiority(*rkn["reconsider"], *rkn["control_T12"], margin=0.01)
    gate.check("reconsider@T8 not worse than plain control@T12 (NI 1pt)", ok,
               d + " — if the T12 control wins, thinking LONGER suffices; say so")
    gate.check("reconsider actually fires (not collapsed)", 0.05 <= n_rec <= 6,
               f"{n_rec:.2f} events/episode")
    ok, d = non_inferiority(*akn["reconsider"], *akn["control"], margin=0.02)
    gate.check("overall accuracy not sacrificed (NI 2pts)", ok, d)
    finish(gate.report())


if __name__ == "__main__":
    main()
