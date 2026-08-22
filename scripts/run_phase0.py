"""Phase 0 — baselines first: A transformer (no recurrence), B full-sequence
recurrent, C slot-recurrent (FAR² topology, zero FAR² operators). C exists so
that D−C isolates the FAR² contribution from the latent-bottleneck advantage.
Run once per model. See docs/ROADMAP.md."""

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.data.registry import CORE_FAMILIES, GENERATORS
from far2.eval import EvalSpec
from far2.eval.metrics import Gate, accuracy
from far2.presets import get_preset
from far2.train import core_mixes, single_stage
from far2.train.curriculum import TRAIN_DMAX


def main():
    p = parser("Phase 0: train a 5M baseline (A/B/C)", steps=4000)
    p.add_argument("--model", required=True,
                   choices=["transformer", "recurrent", "slot_recurrent"])
    args = p.parse_args()
    apply_mini(args)

    mc = get_preset(f"{args.model}_5m")
    depth = "fixed" if args.model == "transformer" else "random"
    stages = single_stage(core_mixes(1.0, hard=True), depth=depth,
                          ipt_alpha=0.0 if args.model == "transformer" else 0.5)
    tc = make_tc(args, f"phase0_{args.model}_seed{args.seed}")
    trainer = train(mc, tc, stages)

    specs = []
    for f in CORE_FAMILIES:
        g = GENERATORS[f]
        ds = ([5, 7, 10] if f == "base_arith"       # trained bases only here
              else sorted({g.min_difficulty, g.min_difficulty + 2,
                           TRAIN_DMAX[f]}))
        specs.append(EvalSpec(f, ds, n_samples=args.eval_samples))
    recs = run_eval(trainer, specs, "phase0")

    gate = Gate(f"Phase 0 · {args.model}_5m")
    for f in CORE_FAMILIES:
        g = GENERATORS[f]
        easy_max = 10 if f == "base_arith" else g.min_difficulty + 2
        easy = accuracy(recs, family=f, difficulty_max=easy_max)
        gate.check(f"{f} easy bucket >= 95%", easy >= 0.95, f"acc={easy:.1%}")
    n_params = trainer.model.param_count()
    gate.check("params within 4.2M-5.8M", 4.2e6 <= n_params <= 5.8e6,
               f"{n_params:,}")
    gate.check("FLOPs logged per episode", all("flops" in r for r in recs),
               f"{len(recs)} records")
    finish(gate.report())


if __name__ == "__main__":
    main()
