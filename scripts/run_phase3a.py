"""Phase 3a (R2) — oracle necessity BEFORE any learned controller: does extra
or split representation actually help when the decisions are PERFECT? Arms:

    fixed8         semantic capacity floor (one concept slot per pair)
    fixed14        capacity ceiling (8 base concepts + max R=6 splits)
    oracle_expand  EXPAND forced by the task condition (perfect timing)
    oracle_split   SPLIT forced by the task condition (perfect timing+target)

The quartet runs on the representation-sensitive mathematics (symmetry
breaking + linear systems). A separately reported `transformer_ref` trains on the identical task
mixture, step budget, batch, seed, and evaluation cells.  It is not a fifth
quartet arm: it diagnoses whether failure belongs to the benchmark/training
budget or specifically to the recurrent workspace foundation.

If fixed8 ~ fixed14 ~ oracle, the benchmark does not require
adaptive representation and a controller would be CORRECT never to split —
that verdict stops the phase. --substrate bind runs the same quartet on the
explicit-binding workspace (R5 re-test). See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec
from far2.eval.metrics import (Gate, accuracy, binding_separation,
                               correct_counts, mean, split_selectivity,
                               token_accuracy)
from far2.eval.stats import non_inferiority, superiority
from far2.presets import get_preset
from far2.train import Stage, core_mixes, fam

FLOOR_CONCEPT_SLOTS = 8
CEILING_CONCEPT_SLOTS = 14
READINESS_EXACT = 0.20
READINESS_TOKEN = 0.75


def base_mc(substrate: str):
    # Every quartet arm carries the same dormant mechanism modules. Fixed arms
    # simply receive no unlocks, while oracle arms change only the decision
    # policy. This makes the comparison parameter-identical, not merely close.
    name = "far2_bind" if substrate == "bind" else "far2_full"
    return replace(get_preset(name), n_iters_train=8)


def mix():
    return core_mixes(0.4) + [fam("linsys", 0.6), fam("symbreak", 1.5)]


def stages_for(arm: str, substrate: str = "standard"):
    if arm == "transformer_ref":
        # Same examples and fixed step budget, without FAR2-only IPT or
        # binding supervision.  The reference therefore gets no hidden
        # auxiliary-label advantage.
        return [Stage("only", 1.0, mix(), depth="fixed",
                      ipt_alpha=0.0, bind_weight=0.0)]
    bw = 0.5 if substrate == "bind" else 0.0
    unlocked = ({"expand", "split"} if arm.startswith("oracle") else set())
    # Fixed T=8 gives the oracle enough action opportunities to service the
    # maximum six broken pairs.  Random T could truncate the upper bound and
    # make operator failure indistinguishable from an insufficient schedule.
    return [Stage("only", 1.0, mix(), unlocked=unlocked, depth="fixed",
                  ipt_alpha=0.5, bind_weight=bw)]


def eval_specs(n):
    return [EvalSpec("symbreak", [1, 2, 4, 6], n_samples=n),
            EvalSpec("linsys", [3, 6], n_samples=n)]


def main():
    p = parser("Phase 3a: oracle-necessity quartet (perfect decisions)",
               steps=4000)
    p.add_argument("--substrate", choices=["standard", "bind"],
                   default="standard",
                   help="bind = standard content attention + registers + an "
                        "explicit contextual-entity/binding substrate")
    args = p.parse_args()
    apply_mini(args)

    base = base_mc(args.substrate)
    tagged = "" if args.substrate == "standard" else "_bind"
    arms = {
        "transformer_ref": get_preset("transformer_5m"),
        "fixed8": replace(base, init_slots=FLOOR_CONCEPT_SLOTS),
        "fixed14": replace(base, init_slots=CEILING_CONCEPT_SLOTS),
        "oracle_expand": replace(base, adaptive_slots=True,
                                 init_slots=FLOOR_CONCEPT_SLOTS,
                                 repr_policy="oracle", oracle_action="expand"),
        "oracle_split": replace(base, adaptive_slots=True,
                                init_slots=FLOOR_CONCEPT_SLOTS,
                                repr_policy="oracle", oracle_action="split"),
    }

    def run_arm(arm):
        mc = arms[arm]
        tc = make_tc(args, f"phase3a_{arm}{tagged}_seed{args.seed}",
                     ipt_alpha=0.0 if arm == "transformer_ref" else 0.5)
        t = train(mc, tc, stages_for(arm, args.substrate))
        return run_eval(t, eval_specs(args.eval_samples), "phase3a")

    # The strongest fixed-capacity foundation is an isolated prerequisite.
    # Do not spend three more arm-runs comparing operators to a model that
    # cannot solve the benchmark in the first place.
    recs = {"fixed14": run_arm("fixed14"),
            "transformer_ref": run_arm("transformer_ref")}
    # v0.4.52: competence is fresh in-distribution generalization, not a blend
    # with R=6 extrapolation beyond the training frontier R<=4.  OOD remains
    # visible below, but cannot make the substrate look untrainable before the
    # representation experiment even begins.
    ceiling_exact = accuracy(
        recs["fixed14"], family="symbreak", requires_distinction=1,
        difficulty_max=4)
    ceiling_token = token_accuracy(
        recs["fixed14"], family="symbreak", requires_distinction=1,
        difficulty_max=4)
    ceiling_ood_exact = accuracy(
        recs["fixed14"], family="symbreak", requires_distinction=1,
        difficulty=6)
    ceiling_ood_token = token_accuracy(
        recs["fixed14"], family="symbreak", requires_distinction=1,
        difficulty=6)
    transformer_exact = accuracy(
        recs["transformer_ref"], family="symbreak", requires_distinction=1,
        difficulty_max=4)
    transformer_token = token_accuracy(
        recs["transformer_ref"], family="symbreak", requires_distinction=1,
        difficulty_max=4)
    transformer_ood_exact = accuracy(
        recs["transformer_ref"], family="symbreak", requires_distinction=1,
        difficulty=6)
    transformer_ood_token = token_accuracy(
        recs["transformer_ref"], family="symbreak", requires_distinction=1,
        difficulty=6)
    far2_flops = mean(recs["fixed14"], "flops", family="symbreak",
                      requires_distinction=1)
    transformer_flops = mean(
        recs["transformer_ref"], "flops", family="symbreak",
        requires_distinction=1)
    flop_ratio = (far2_flops / transformer_flops
                  if transformer_flops > 0 else float("nan"))
    print("\n  paired foundation comparison:")
    print(f"    fixed14:    exact {ceiling_exact:.1%} "
          f"(need >= {READINESS_EXACT:.0%}) · token {ceiling_token:.1%} "
          f"(need >= {READINESS_TOKEN:.0%}) · inference FLOPs {far2_flops:,.0f}")
    print(f"    Transformer: exact {transformer_exact:.1%} · "
          f"token {transformer_token:.1%} · inference FLOPs "
          f"{transformer_flops:,.0f} · FAR2/Transformer {flop_ratio:.2f}x")
    print(f"    OOD R=6 (reported, not pooled into readiness): fixed14 "
          f"exact {ceiling_ood_exact:.1%} · token {ceiling_ood_token:.1%}; "
          f"Transformer exact {transformer_ood_exact:.1%} · "
          f"token {transformer_ood_token:.1%}")
    ready = (ceiling_exact >= READINESS_EXACT
             and ceiling_token >= READINESS_TOKEN)
    if not ready:
        gate = Gate(f"Phase 3a · isolated foundation readiness "
                    f"({args.substrate} substrate)")
        gate.check("capacity ceiling is competent before arm comparison",
                   False, f"NOT-READY: fixed14 exact={ceiling_exact:.1%}, "
                          f"token={ceiling_token:.1%}; require exact>="
                          f"{READINESS_EXACT:.0%} and token>={READINESS_TOKEN:.0%}")
        gate.check("Transformer reference recorded on the same task budget",
                   True, f"exact={transformer_exact:.1%}, "
                         f"token={transformer_token:.1%}, "
                         f"FAR2/Transformer FLOPs={flop_ratio:.2f}x")
        finish(gate.report())

    for arm in ("fixed8", "oracle_expand", "oracle_split"):
        recs[arm] = run_arm(arm)

    order = ["fixed8", "oracle_expand", "oracle_split", "fixed14"]
    div = {a: accuracy(r, family="symbreak", requires_distinction=1)
           for a, r in recs.items()}
    kn = {a: correct_counts(r, family="symbreak", requires_distinction=1)
          for a, r in recs.items()}
    sel = split_selectivity(recs["oracle_split"])
    bind = binding_separation(recs["oracle_split"])
    oracle_dosage = all(
        (r["split_count"] == float(r["difficulty"])
         if r["requires_distinction"] == 1 else r["split_count"] == 0.0)
        for r in recs["oracle_split"] if r["family"] == "symbreak")

    print("\n           acc(distinction queries)   [oracle = ground-truth-forced]")
    for a in order:
        print(f"  {a:<14} {div[a]:>10.1%}")
    print(f"  oracle split fires: when-required {sel['rate_when_required']:.1%} "
          f"· on-control {sel['rate_on_control']:.1%}")
    print(f"  binding: separated-when-required "
          f"{bind.get('separated_when_required', float('nan')):.1%} · "
          f"after-split {bind.get('separated_after_split', float('nan')):.1%}")
    print(f"  readiness ceiling: exact {ceiling_exact:.1%} "
          f"(need >= {READINESS_EXACT:.0%}) · token {ceiling_token:.1%} "
          f"(need >= {READINESS_TOKEN:.0%})")

    gate = Gate(f"Phase 3a · oracle necessity ({args.substrate} substrate)")
    gate.check("Transformer reference reported (not a quartet arm)", True,
               f"exact={transformer_exact:.1%}, token={transformer_token:.1%}, "
               f"FAR2/Transformer FLOPs={flop_ratio:.2f}x")
    gate.check("oracle SPLIT dosage matches every broken pair R",
               oracle_dosage, "split_count=R on distinction episodes; 0 on controls")
    ok, d = superiority(*kn["fixed14"], *kn["fixed8"], margin=0.05)
    gate.check("the benchmark REQUIRES representation (fixed14 > fixed8 +5pts)",
               ok, d + " — if this fails, a controller is CORRECT never to "
               "split; fix the benchmark before training one")
    ok, d = superiority(*kn["oracle_split"], *kn["fixed8"], margin=0.05)
    gate.check("perfect SPLIT beats the capacity floor (+5pts)", ok, d)
    ok, d = non_inferiority(*kn["oracle_split"], *kn["fixed14"], margin=0.02)
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
