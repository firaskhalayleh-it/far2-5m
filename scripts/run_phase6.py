"""Phase 6 — MATHEMATICAL epistemic state on linear systems: DERIVED (unique
solution) vs UNDERDETERMINED vs NO_SOLUTION. There is provably no justified
numeric answer in the non-unique cases, so a digit there is a genuine
unsupported assertion — and refusing solvable systems is the mirror failure.
See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec
from far2.eval.metrics import Gate, accuracy, assertion_counts, assertion_rates
from far2.eval.stats import non_inferiority, rate_at_most
from far2.presets import get_preset
from far2.train import core_mixes, fam, single_stage


def main():
    p = parser("Phase 6: mathematical epistemic state vs ablation", steps=3000)
    args = p.parse_args()
    apply_mini(args)

    mix = core_mixes(0.3) + [fam("symbreak", 0.5), fam("linsys", 1.5)]
    arms = {"epistemic": replace(get_preset("far2_5m"), epistemic=True,
                                 init_slots=8, max_slots=8),
            "control": replace(get_preset("far2_5m"), init_slots=8, max_slots=8)}
    recs = {}
    for arm, mc in arms.items():
        unlocked = {"epistemic"} if arm == "epistemic" else set()
        tc = make_tc(args, f"phase6_{arm}_seed{args.seed}")
        t = train(mc, tc, single_stage(mix, unlocked=unlocked, depth="random",
                                       ipt_alpha=0.5))
        recs[arm] = run_eval(t, [EvalSpec("linsys", [3, 5, 8],
                                          n_samples=args.eval_samples)], "phase6")

    rates = {a: assertion_rates(r) for a, r in recs.items()}
    answerable = {a: accuracy([x for x in r if x.get("is_unknown") is False])
                  for a, r in recs.items()}
    by_status = {st: accuracy(recs["epistemic"], status=st)
                 for st in ("unique", "under", "inconsistent")}

    print("\n              unsupported  over-abstain  answerable-acc")
    for a in arms:
        print(f"  {a:<10} {rates[a][0]:>10.1%} {rates[a][1]:>12.1%} "
              f"{answerable[a]:>13.1%}")
    print("  epistemic arm by status: " + " · ".join(
        f"{k} {v:.1%}" for k, v in by_status.items()))

    (ku, nu), (ko, na) = assertion_counts(recs["epistemic"])
    gate = Gate("Phase 6 · mathematical epistemic state")
    ok, d = rate_at_most(ku, nu, point_max=0.10, ci_ceiling=0.20)
    gate.check("unsupported assertion rate <= 10%", ok,
               d + f" (control {rates['control'][0]:.1%})")
    ok, d = rate_at_most(ko, na, point_max=0.10, ci_ceiling=0.20)
    gate.check("over-abstention <= 10% (refusing everything = failure)", ok, d)
    from far2.eval.metrics import correct_counts
    ka = correct_counts([x for x in recs["epistemic"]
                         if x.get("is_unknown") is False])
    kc = correct_counts([x for x in recs["control"]
                         if x.get("is_unknown") is False])
    ok, d = non_inferiority(*ka, *kc, margin=0.02)
    gate.check("answerable accuracy within 2pts of control (NI)", ok, d)
    finish(gate.report())


if __name__ == "__main__":
    main()
