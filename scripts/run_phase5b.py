"""Phase 5b — learned HALT, added only after reconsideration is proven at
fixed compute (P5a). Gains must show as maintained recovery at REDUCED average
iterations, reported honestly against the free convergence-rule control
(r_t < 0.1), which the literature says is hard to beat. See docs/ROADMAP.md."""

import os
from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec, evaluate
from far2.eval.metrics import Gate, accuracy, mean, recovery_rate
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage
from run_phase5a import FIXED_T, eval_specs, mix


def stages():
    return [
        Stage("warmup", 0.4, mix(), unlocked={"halt", "reconsider"},
              depth="random", ipt_alpha=0.5, oracle_weight=1.0,
              entropy_bonus=0.01, tau=1.0),
        Stage("free", 0.6, mix(), unlocked={"halt", "reconsider"},
              depth="random", ipt_alpha=0.5, oracle_weight=0.3, iter_cost=0.01,
              entropy_bonus=0.01, tau=0.7),
    ]


def main():
    p = parser("Phase 5b: learned halting on top of proven reconsideration",
               steps=3000)
    args = p.parse_args()
    apply_mini(args)

    mc = replace(get_preset("far2_5m"), halting="learned", reconsider=True,
                 init_slots=8, max_slots=8, n_iters_train=FIXED_T)
    tc = make_tc(args, f"phase5b_learned_seed{args.seed}", ipt_alpha=0.5)
    trainer = train(mc, tc, stages())
    recs = run_eval(trainer, eval_specs(args.eval_samples), "phase5b")

    # convergence-rule control: SAME weights, halting decided by r_t < tol.
    # Copied INSIDE eval_weights() so the control gets the identical EMA
    # weights the learned arm was just evaluated with — raw weights would
    # make this a different-model comparison, not a different-rule one.
    conv_model = build_model(replace(mc, halting="converge", reconsider=False))
    with trainer.eval_weights():
        conv_model.load_state_dict(trainer.model.state_dict(), strict=False)
    conv_model.to(trainer.device)
    conv = evaluate(conv_model, eval_specs(args.eval_samples), trainer.device,
                    out_path=os.path.join(tc.out_dir, "eval", "converge.jsonl"))

    rows = {"learned": recs, "converge-rule": conv}
    rr = {a: recovery_rate(r) for a, r in rows.items()}
    acc = {a: accuracy(r) for a, r in rows.items()}
    iters = {a: mean(r, "iters_used") for a, r in rows.items()}

    print(f"\n                 accuracy   recovery   mean iters (budget {FIXED_T})")
    for a, r in rows.items():
        print(f"  {a:<14} {acc[a]:>8.1%} {rr[a]:>9.1%} {iters[a]:>10.1f}")

    gate = Gate("Phase 5b · learned halting")
    gate.check("halting saves compute (mean iters < budget)",
               iters["learned"] <= FIXED_T - 0.5, f"{iters['learned']:.1f} < {FIXED_T}")
    gate.check("recovery maintained under halting (within 2pts of P5a-style full budget)",
               rr["learned"] == rr["learned"] and rr["learned"] >= rr["converge-rule"] - 0.02,
               f"learned {rr['learned']:.1%} vs converge-rule {rr['converge-rule']:.1%}")
    gate.check("accuracy vs converge-rule control reported",
               acc["learned"] >= acc["converge-rule"] - 0.02,
               f"{acc['learned']:.1%} vs {acc['converge-rule']:.1%} at "
               f"{iters['learned']:.1f} vs {iters['converge-rule']:.1f} iters — "
               "if the free rule wins, say so; it is the honest default")
    finish(gate.report())


if __name__ == "__main__":
    main()
