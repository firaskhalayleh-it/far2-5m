"""Phase 3b (R7) — FREE adaptive representation controller, using the signal
the Phase 4 tournament selected, embedded in the necessity-vs-allocation
quartet on mathematical symmetry breaking:

    fixed4     capacity floor            (4 slots forever)
    random     rate-matched RANDOM expansion/splitting, random split source
    fixed16    capacity ceiling          (16 slots forever)
    adaptive   the learned controller

Reading the quartet (pre-registered):
  · random closes most of the fixed16-fixed4 gap  -> the win is capacity/noise
  · adaptive ~ fixed16 at <=70% slot-compute while random ~ fixed4
                                                  -> intelligent allocation
  · adaptive > fixed16 while random ~ fixed4      -> restructuring beyond
                                                     capacity (the big claim)
Plus the surface gate: ACTIVE SLOTS must track representation demand R
(broken pairs), not surface length L. The random arm's rates are matched to
the trained adaptive arm's measured firing rates. Controls are reused from
Phase 3a when compatible runs exist. docs/ROADMAP.md."""

import json
import os
from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval.metrics import (Gate, accuracy, binding_separation,
                               correct_counts, mean, split_selectivity,
                               surface_slopes)
from far2.eval.stats import non_inferiority, rate_at_least, superiority
from far2.train import Stage
from far2.utils.logging import read_jsonl
from run_phase3a import base_mc, eval_specs, mix, stages_for


def free_stages(substrate: str):
    # supervision MUST reach zero before any claim: the evaluated checkpoint's
    # final stage has oracle_weight=0 and bind_weight=0 (asserted by tests)
    bw = 0.5 if substrate == "bind" else 0.0
    return [
        Stage("warmup", 0.3, mix(), unlocked={"expand"}, depth="random",
              ipt_alpha=0.5, oracle_weight=1.0, bind_weight=bw,
              entropy_bonus=0.01, tau=1.0),
        Stage("free", 0.4, mix(), unlocked={"expand", "split"}, depth="random",
              ipt_alpha=0.5, oracle_weight=0.3, bind_weight=bw / 2,
              slot_cost=0.01, entropy_bonus=0.01, tau=0.7),
        Stage("autonomous", 0.3, mix(), unlocked={"expand", "split"},
              depth="random", ipt_alpha=0.5, oracle_weight=0.0,
              bind_weight=0.0, slot_cost=0.01, entropy_bonus=0.005, tau=0.6),
    ]


def random_stages(substrate: str):
    """IDENTICAL stages to the adaptive arm — same data mixes, fractions,
    depth schedule, oracle/bind loss weights, slot cost, entropy, tau. The
    ONLY difference between the arms is repr_policy='random' (decisions
    replaced by rate-matched noise). Anything less is an unmatched control:
    an adaptive win could be blamed on curriculum or loss-weight differences
    (review blocker: 'random arm trains on a different curriculum')."""
    return free_stages(substrate)


def control_records(args, arm, base, tagged):
    path = os.path.join(args.out, f"phase3a_{arm}{tagged}_seed{args.seed}",
                        "eval", "phase3a.jsonl")
    if os.path.exists(path):
        print(f"[phase3b] reusing {arm} control from {path}")
        return read_jsonl(path)
    mc = replace(base, init_slots=(16 - base.n_registers if arm == "fixed16"
                                   else 4))
    tc = make_tc(args, f"phase3b_{arm}{tagged}_seed{args.seed}", ipt_alpha=0.5)
    t = train(mc, tc, stages_for(arm, args.substrate))
    return run_eval(t, eval_specs(args.eval_samples), "phase3b")


def interpret(acc4, accR, acc16, accA, scA, sc16) -> str:
    gap = acc16 - acc4
    if gap <= 0.02:
        return ("capacity barely matters on this eval (fixed16-fixed4 gap "
                f"{gap:+.1%}) — the quartet cannot separate the hypotheses")
    r_frac = (accR - acc4) / gap
    if accA >= acc16 + 0.03 and r_frac <= 0.4:
        return ("RESTRUCTURING BEYOND CAPACITY: adaptive beats even fixed16 "
                "while random stays near the floor")
    if accA >= acc16 - 0.01 and scA <= 0.70 * sc16 and r_frac <= 0.4:
        return ("INTELLIGENT ALLOCATION: adaptive matches full capacity at "
                f"{scA / sc16:.0%} of its slot-compute while random expansion "
                "does not — the decisions carry the value")
    if r_frac >= 0.6:
        return (f"CAPACITY/NOISE STORY: random expansion closes {r_frac:.0%} "
                "of the capacity gap — little evidence the learned decisions "
                "matter; report any adaptive win as capacity, not cognition")
    return "MIXED: between the reference patterns — report all four numbers"


def main():
    p = parser("Phase 3b: free adaptive controller + necessity-vs-allocation "
               "quartet", steps=4000)
    p.add_argument("--signal", default=None,
                   help="collision signal (default: runs/selected_signal.json, "
                        "else substrate default)")
    p.add_argument("--substrate", choices=["standard", "bind"],
                   default="standard")
    args = p.parse_args()
    apply_mini(args)

    base = base_mc(args.substrate)
    tagged = "" if args.substrate == "standard" else "_bind"
    signal = args.signal
    sel_path = os.path.join(args.out, "selected_signal.json")
    if signal is None and os.path.exists(sel_path):
        sel = json.load(open(sel_path))
        if sel.get("substrate", args.substrate) != args.substrate:
            print(f"[phase3b] ignoring {sel_path}: selected on substrate="
                  f"{sel.get('substrate')} != {args.substrate}")
        else:
            signal = sel["signal"]
    if signal is None:
        # a silent fallback would train the headline arm on an unvalidated
        # signal after a FAILED tournament and report it as if selected
        msg = (f"[phase3b] no Phase 4 selection for substrate="
               f"{args.substrate} ({sel_path} absent or mismatched) and no "
               "--signal given.")
        if args.mini:
            signal = base.collision_signal
            print(msg + f" MINI wiring mode only: falling back to substrate "
                        f"default '{signal}'.")
        else:
            raise SystemExit(msg + " REFUSING to train — run run_phase4.py "
                             "first or pass --signal explicitly.")
    print(f"[phase3b] substrate={args.substrate} · controller signal: {signal}")

    mc = replace(base, adaptive_slots=True, init_slots=4,
                 collision_signal=signal)
    tc = make_tc(args, f"phase3b_adaptive{tagged}_seed{args.seed}", ipt_alpha=0.5)
    trainer = train(mc, tc, free_stages(args.substrate))
    recs = {"adaptive": run_eval(trainer, eval_specs(args.eval_samples), "phase3b")}

    # random-policy control: same architecture, growth rates matched to what
    # the adaptive arm actually did — decisions replaced by noise
    # NO lower clamp: if the adaptive arm never expands, a matched random
    # arm must never expand — a 0.005 floor fakes matching at zero usage
    iters = max(mean(recs["adaptive"], "iters_used"), 1.0)
    e_rate = min(mean(recs["adaptive"], "expand_count") / iters, 1.0)
    s_rate = min(mean(recs["adaptive"], "split_count") / iters, 1.0)
    print(f"[phase3b] random arm rates matched to adaptive (raw, no floor): "
          f"P(EXPAND)={e_rate:.4f}/iter · P(SPLIT)={s_rate:.4f}/iter")
    mc_rand = replace(mc, repr_policy="random",
                      random_expand_p=e_rate, random_split_p=s_rate)
    tc_rand = make_tc(args, f"phase3b_random{tagged}_seed{args.seed}",
                      ipt_alpha=0.5)
    t_rand = train(mc_rand, tc_rand, random_stages(args.substrate))
    recs["random"] = run_eval(t_rand, eval_specs(args.eval_samples), "phase3b")

    recs["fixed16"] = control_records(args, "fixed16", base, tagged)
    recs["fixed4"] = control_records(args, "fixed4", base, tagged)

    def slot_compute(rs):
        return mean(rs, "active_slots") * mean(rs, "iters_used")

    order = ["fixed4", "random", "fixed16", "adaptive"]
    kn = {a: correct_counts(r, family="symbreak") for a, r in recs.items()}
    div = {a: accuracy(r, family="symbreak", requires_distinction=1)
           for a, r in recs.items()}
    kn_div = {a: correct_counts(r, family="symbreak", requires_distinction=1)
              for a, r in recs.items()}
    sc = {a: slot_compute(r) for a, r in recs.items()}
    sel = split_selectivity(recs["adaptive"])
    bind = binding_separation(recs["adaptive"])
    n_active = mean(recs["adaptive"], "active_slots")
    sym_ad = [r for r in recs["adaptive"] if r["family"] == "symbreak"]
    ss = surface_slopes(sym_ad, "active_slots", "R", "L")

    print("\n  necessity-vs-allocation quartet (symbreak)")
    print("            acc(all)  acc(distinction-q)  slot·iters")
    for a in order:
        print(f"  {a:<9} {kn[a][0] / max(kn[a][1], 1):>8.1%} {div[a]:>16.1%} "
              f"{sc[a]:>11.1f}")
    cap_gap = div["fixed16"] - div["fixed4"]
    R = (div["random"] - div["fixed4"]) / cap_gap if cap_gap > 0 else float("nan")
    rho = sc["adaptive"] / sc["fixed16"] if sc["fixed16"] else float("nan")
    print(f"  indices: capacity gap Δ16−4 = {cap_gap:+.1%} · random-closure "
          f"R = {R:.2f} · compute ratio ρ = {rho:.2f}")
    print(f"  reading: "
          f"{interpret(div['fixed4'], div['random'], div['fixed16'], div['adaptive'], sc['adaptive'], sc['fixed16'])}")
    if ss.get("testable"):
        print(f"  surface: slots ~ {ss['beta1']:+.2f}·R {ss['beta2']:+.2f}·L "
              f"(z_R={ss['z1']:.2f} p={ss['p1_pos']:.4f}; "
              f"z_L={ss['z2']:.2f} p={ss['p2_pos']:.4f})")

    gate = Gate("Phase 3b · dynamic representation (free controller)")
    ok, d = non_inferiority(*kn["adaptive"], *kn["fixed16"], margin=0.01)
    gate.check("adaptive not worse than fixed16 on symbreak (NI 1pt)", ok, d)
    gate.check("adaptive uses <= 70% of fixed16 slot-compute (rho <= 0.70)",
               rho <= 0.70, f"rho = {rho:.2f} "
               f"({sc['adaptive']:.1f} vs {sc['fixed16']:.1f} slot·iters)")
    ok, d = superiority(*kn_div["adaptive"], *kn_div["random"], margin=0.03)
    gate.check("adaptive > rate-matched RANDOM policy (distinction queries)",
               ok, d + " — else the win is injected capacity/noise")
    ok_p, d_p = rate_at_least(sel["tp"], sel["n_did"],
                              point_min=0.60, ci_floor=0.50)
    ok_r, d_r = rate_at_least(sel["tp"], sel["n_need"],
                              point_min=0.50, ci_floor=0.40)
    gate.check("split precision vs condition (>=0.60, CI floor 0.50)", ok_p, d_p)
    gate.check("split recall vs condition (>=0.50, CI floor 0.40)", ok_r, d_r)
    gate.check("no controller collapse", 3.5 <= n_active <= 15.5,
               f"mean active slots {n_active:.1f}")
    slots_ok = (ss.get("testable", False) and ss["p1_pos"] < 0.05
                and (ss["p2_pos"] >= 0.05 or ss["beta2"] <= 0.5 * ss["beta1"]))
    gate.check("active slots track R (demand), not L (length)", slots_ok,
               "surface regression above — 'many equations -> more slots' "
               "would show as beta_L, not beta_R")
    ok, d = superiority(bind.get("k_split", 0), bind.get("n_split", 0),
                        bind.get("k_ctrl", 0), bind.get("n_ctrl", 0),
                        margin=0.0)
    gate.check("binding separation after split > control rate", ok,
               d + " — else SPLIT is generic capacity allocation "
               "(different claim)")
    print("\n  next: run_interventions.py --run "
          f"{tc.out_dir}  (usage gates, then episode-level do())\n"
          "        run_probes.py --run "
          f"{tc.out_dir} --control runs/phase3a_fixed16{tagged}_seed{args.seed}")
    finish(gate.report())


if __name__ == "__main__":
    main()
