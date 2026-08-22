"""Phase 3a (R2) — oracle necessity BEFORE any learned controller: does extra
or split representation actually help when the decisions are PERFECT? Arms:

    fixed4         capacity floor
    fixed16        capacity ceiling
    oracle_expand  EXPAND forced by the task condition (perfect timing)
    oracle_split   SPLIT forced by the task condition (perfect timing+target)

on the representation-sensitive mathematics (symmetry breaking + linear
systems). If fixed4 ~ fixed16 ~ oracle, the benchmark does not require
adaptive representation and a controller would be CORRECT never to split —
that verdict stops the phase. --substrate bind runs the same quartet on the
competitive-binding workspace (R5 re-test). See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec
from far2.eval.metrics import (Gate, accuracy, binding_separation,
                               correct_counts, split_selectivity)
from far2.eval.stats import non_inferiority, superiority
from far2.presets import get_preset
from far2.train import Stage, core_mixes, fam


def base_mc(substrate: str):
    mc = replace(get_preset("far2_5m"), n_iters_train=8)
    if substrate == "bind":
        mc = replace(mc, cross_mode="competitive", n_registers=2, binding=True,
                     collision_signal="bind")
    return mc


def mix():
    return core_mixes(0.4) + [fam("linsys", 0.6), fam("symbreak", 1.5)]


def stages_for(arm: str, substrate: str = "standard"):
    bw = 0.5 if substrate == "bind" else 0.0
    unlocked = ({"expand", "split"} if arm.startswith("oracle") else set())
    return [Stage("only", 1.0, mix(), unlocked=unlocked, depth="random",
                  ipt_alpha=0.5, bind_weight=bw)]


def eval_specs(n):
    return [EvalSpec("symbreak", [1, 2, 4, 6], n_samples=n),
            EvalSpec("linsys", [3, 6], n_samples=n)]


def main():
    p = parser("Phase 3a: oracle-necessity quartet (perfect decisions)",
               steps=4000)
    p.add_argument("--substrate", choices=["standard", "bind"],
                   default="standard",
                   help="bind = competitive attention + registers + binding "
                        "matrix (R5 re-test after the substrate revision)")
    args = p.parse_args()
    apply_mini(args)

    base = base_mc(args.substrate)
    tagged = "" if args.substrate == "standard" else "_bind"
    arms = {
        "fixed4": replace(base, init_slots=4),
        "fixed16": replace(base, init_slots=16 - base.n_registers),
        "oracle_expand": replace(base, adaptive_slots=True, init_slots=4,
                                 repr_policy="oracle", oracle_action="expand"),
        "oracle_split": replace(base, adaptive_slots=True, init_slots=4,
                                repr_policy="oracle", oracle_action="split"),
    }
    recs = {}
    for arm, mc in arms.items():
        tc = make_tc(args, f"phase3a_{arm}{tagged}_seed{args.seed}",
                     ipt_alpha=0.5)
        t = train(mc, tc, stages_for(arm, args.substrate))
        recs[arm] = run_eval(t, eval_specs(args.eval_samples), "phase3a")

    order = ["fixed4", "oracle_expand", "oracle_split", "fixed16"]
    div = {a: accuracy(r, family="symbreak", requires_distinction=1)
           for a, r in recs.items()}
    kn = {a: correct_counts(r, family="symbreak", requires_distinction=1)
          for a, r in recs.items()}
    sel = split_selectivity(recs["oracle_split"])
    bind = binding_separation(recs["oracle_split"])

    print("\n           acc(distinction queries)   [oracle = ground-truth-forced]")
    for a in order:
        print(f"  {a:<14} {div[a]:>10.1%}")
    print(f"  oracle split fires: when-required {sel['rate_when_required']:.1%} "
          f"· on-control {sel['rate_on_control']:.1%}")
    print(f"  binding: separated-when-required "
          f"{bind.get('separated_when_required', float('nan')):.1%} · "
          f"after-split {bind.get('separated_after_split', float('nan')):.1%}")

    gate = Gate(f"Phase 3a · oracle necessity ({args.substrate} substrate)")
    ok, d = superiority(*kn["fixed16"], *kn["fixed4"], margin=0.05)
    gate.check("the benchmark REQUIRES representation (fixed16 > fixed4 +5pts)",
               ok, d + " — if this fails, a controller is CORRECT never to "
               "split; fix the benchmark before training one")
    ok, d = superiority(*kn["oracle_split"], *kn["fixed4"], margin=0.05)
    gate.check("perfect SPLIT beats the capacity floor (+5pts)", ok, d)
    ok, d = non_inferiority(*kn["oracle_split"], *kn["fixed16"], margin=0.02)
    gate.check("perfect SPLIT reaches the capacity ceiling (NI 2pts)", ok, d)
    ok, d = superiority(*kn["oracle_split"], *kn["oracle_expand"], margin=0.0)
    gate.check("targeted SPLIT vs generic EXPAND reported", True,
               d + (" — split adds nothing beyond generic capacity here"
                    if not ok else " — targeting matters"))
    gate.check("binding diagnostics recorded (claim-prerequisite for SPLIT)",
               bind.get("n", 0) > 0, f"n={bind.get('n', 0)}")
    finish(gate.report())


if __name__ == "__main__":
    main()
