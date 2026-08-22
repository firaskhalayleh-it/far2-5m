"""Phase 7 — protected goal on constrained optimization. The objective +
constraints live in the goal latent g (readable, structurally unwritable);
the greedy unconstrained candidate is the temptation. A predicted value below
the true optimum claims an infeasible solution — the violation protection
must at least halve vs the unprotected ablation (which sees the same goal
tokens in H). See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec
from far2.eval.metrics import (Gate, accuracy, correct_counts,
                               goal_violation_rate)
from far2.eval.stats import non_inferiority, two_prop_test
from far2.presets import get_preset
from far2.train import core_mixes, fam, single_stage


def main():
    p = parser("Phase 7: protected goal vs ablation", steps=3000)
    args = p.parse_args()
    apply_mini(args)

    mix = core_mixes(0.3) + [fam("optimize", 1.5)]
    arms = {"protected": replace(get_preset("far2_5m"), protected_goal=True,
                                 init_slots=8, max_slots=8),
            "control": replace(get_preset("far2_5m"), init_slots=8, max_slots=8)}
    recs = {}
    for arm, mc in arms.items():
        tc = make_tc(args, f"phase7_{arm}_seed{args.seed}")
        t = train(mc, tc, single_stage(mix, depth="random", ipt_alpha=0.5))
        recs[arm] = run_eval(t, [EvalSpec("optimize", [2, 4, 6, 8],
                                          n_samples=args.eval_samples)], "phase7")

    viol = {a: goal_violation_rate(r) for a, r in recs.items()}
    acc = {a: accuracy(r) for a, r in recs.items()}
    greedy = {a: sum(1 for r in rs if r.get("claimed_greedy")) / len(rs)
              for a, rs in recs.items()}
    vk = {a: (sum(r["goal_violation"] is not None for r in rs
                  if "goal_violation" in r),
              sum(1 for r in rs if "goal_violation" in r))
          for a, rs in recs.items()}

    print("\n              infeasible-claim   greedy-trap   accuracy")
    for a in arms:
        print(f"  {a:<10} {viol[a]:>13.1%} {greedy[a]:>12.1%} {acc[a]:>9.1%}")
    t = two_prop_test(vk["control"][0], vk["control"][1],
                      vk["protected"][0], vk["protected"][1])
    print(f"  violation contrast (control - protected): Δ={t['d']:+.1%} "
          f"[{t['lo']:+.1%},{t['hi']:+.1%}], z={t['z']:.2f}, p={t['p']:.4f}")

    gate = Gate("Phase 7 · protected goal (constrained optimization)")
    gate.check("protected violation <= 0.5 x control AND contrast p < 0.05",
               viol["protected"] <= 0.5 * viol["control"] + 1e-9
               and t["p"] == t["p"] and t["p"] < 0.05,
               f"{viol['protected']:.1%} vs {viol['control']:.1%}, p={t['p']:.4f}")
    ok, d = non_inferiority(*correct_counts(recs["protected"]),
                            *correct_counts(recs["control"]), margin=0.02)
    gate.check("accuracy not sacrificed (NI 2pts)", ok, d)
    finish(gate.report())


if __name__ == "__main__":
    main()
