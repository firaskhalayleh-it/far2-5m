"""Operator causal interventions (R8+R9). Training-time ablation says "a model
without X is worse"; this says "THIS decision, in THIS episode, caused THIS
answer". Per operator, TWO stages with four-state verdicts:

R8 usage prerequisite — before any do(), the operator must actually be used:
      UsageRate      = P(action | condition present)   gate >= 15%
      FalseUsageRate = P(action | condition absent)    gate <=  5%
   'condition present = 500, used = 3' is not "causal test unavailable" — it
   is already the finding USAGE FAILURE: the controller did not learn to use
   the operator. That verdict is reported as such, never lumped with
   NOT-TESTABLE (condition population too small).

R9 do() interventions — episodes replay exactly (eval is deterministic):
   necessity    correct-and-used, operator BLOCKED   -> flip-to-wrong rate
   sufficiency  wrong-and-unused (condition present), FORCED at t=T//2
   placebo      correct-no-condition, FORCED         -> flip-to-wrong rate

Verdict vocabulary: PASS | FAIL | NOT-TESTABLE | USAGE-FAILURE.
Run on a trained FAR² checkpoint (default: the Phase 3b adaptive run)."""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from far2.data.base import collate, randomize_family_token
from far2.data.registry import ALL_GENERATORS
from far2.data.vocab import VOCAB
from far2.eval.metrics import Gate
from far2.eval.stats import (N_MIN, rate_at_least, rate_at_most,
                             two_prop_test)
from far2.train import Trainer
from far2.utils import pick_device

# operator -> (family it should matter for, telemetry count key, task-condition key)
OPS = {
    "split": ("symbreak", "split_count", "requires_distinction"),
    "expand": ("symbreak", "expand_count", "requires_distinction"),
    "reconsider": ("extraneous", "reconsider_count", "reconsider_label"),
}
ALL_UNLOCKS = {"expand", "split", "halt", "reconsider", "epistemic"}


@torch.no_grad()
def run_episode(model, sample, device, T, intervene=None):
    batch = collate([sample], device)
    out = model(batch, n_iters=T, intervene=intervene)
    n_ans = int((batch["target_ids"][0] != -100).sum())
    pred = VOCAB.decode(out.logits.argmax(-1)[0, :n_ans].tolist())
    gen = ALL_GENERATORS[sample.family]
    return bool(gen.score(sample, pred)["correct"]), out.telemetry


def collect_pool(model, gen, device, T, n, difficulty, seed):
    rng = random.Random(f"interv:{seed}:{gen.name}:{difficulty}")
    pool = []
    for _ in range(n):
        s = randomize_family_token(gen.sample(rng, difficulty), rng)
        correct, tel = run_episode(model, s, device, T)
        pool.append((s, correct, {k: tel[k][0] for k in
                                  ("split_count", "expand_count",
                                   "reconsider_count") if k in tel}))
    return pool


def paired_rate(flips: list[bool]) -> float:
    return sum(flips) / len(flips) if flips else float("nan")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Episode-level do() interventions "
                                            "on a trained FAR2 checkpoint")
    p.add_argument("--run", default="runs/phase3b_adaptive_seed0",
                   help="run dir containing ckpt.pt")
    p.add_argument("--ops", nargs="+", default=["split", "expand", "reconsider"],
                   choices=list(OPS))
    p.add_argument("--n", type=int, default=300,
                   help="episode pool size per operator")
    p.add_argument("--difficulty", type=int, default=6)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    device = pick_device(args.device)
    model, cfg = Trainer.load_model(os.path.join(args.run, "ckpt.pt"), device)
    model.eval()
    if hasattr(model, "set_unlocked"):
        model.set_unlocked(ALL_UNLOCKS)
    if getattr(cfg, "repr_policy", "learned") != "learned":
        print(f"[warn] checkpoint repr_policy={cfg.repr_policy}: necessity "
              "numbers describe that policy, not a learned controller")
    print(f"[interventions] {args.run} · params={model.param_count():,} · "
          f"device={device} · T={args.iters}")

    results = {}
    for op in args.ops:
        family, count_key, cond_key = OPS[op]
        if op in ("split", "expand") and not cfg.adaptive_slots:
            print(f"  {op}: skipped (adaptive_slots off in this checkpoint)")
            continue
        if op == "reconsider" and not cfg.reconsider:
            print(f"  {op}: skipped (reconsider off in this checkpoint)")
            continue
        gen = ALL_GENERATORS[family]
        d = max(gen.min_difficulty, min(args.difficulty, gen.max_difficulty))
        pool = collect_pool(model, gen, device, args.iters, args.n, d, args.seed)

        cond = lambda s: getattr(s, cond_key) == 1
        used = lambda c: c.get(count_key, 0) > 0

        # R8 usage prerequisite — natural operator demand
        n_cond = sum(cond(s) for s, _, _ in pool)
        n_ctrl = len(pool) - n_cond
        used_pos = sum(used(c) for s, _, c in pool if cond(s))
        used_neg = sum(used(c) for s, _, c in pool if not cond(s))
        usage = used_pos / n_cond if n_cond else float("nan")
        false_usage = used_neg / n_ctrl if n_ctrl else float("nan")
        if n_cond < N_MIN:
            verdict = "NOT-TESTABLE (condition population too small)"
        elif usage < 0.15:
            verdict = (f"USAGE-FAILURE: condition present {n_cond}x, operator "
                       f"used {used_pos}x — the controller did not learn to "
                       "use it; do() tests skipped")
        else:
            verdict = "usage OK — running do() interventions"
        print(f"\n  {op} on {family} d{d} — pool {len(pool)}")
        print(f"    usage: P(use|cond)={usage:.1%} (n={n_cond}) · "
              f"P(use|no-cond)={false_usage:.1%} (n={n_ctrl})")
        print(f"    verdict: {verdict}")

        results[op] = {
            "family": family, "difficulty": d, "pool": len(pool),
            "n_cond": n_cond, "used_pos": used_pos,
            "n_ctrl": n_ctrl, "used_neg": used_neg,
            "usage": usage, "false_usage": false_usage, "verdict": verdict,
        }
        if not verdict.startswith("usage OK"):
            continue

        nec_pool = [s for s, ok, c in pool if ok and used(c)]
        suf_pool = [s for s, ok, c in pool
                    if not ok and not used(c) and cond(s)]
        pla_pool = [s for s, ok, c in pool if ok and not cond(s)]

        force_t = max(1, args.iters // 2)
        nec_flip = [not run_episode(model, s, device, args.iters,
                                    {"block": {op}})[0] for s in nec_pool]
        suf_flip = [run_episode(model, s, device, args.iters,
                                {"force": [(op, force_t)]})[0] for s in suf_pool]
        pla_flip = [not run_episode(model, s, device, args.iters,
                                    {"force": [(op, force_t)]})[0]
                    for s in pla_pool[:max(len(nec_pool), 40) or None]]

        results[op].update({
            "necessity_n": len(nec_pool), "necessity_flip": paired_rate(nec_flip),
            "necessity_k": sum(nec_flip),
            "sufficiency_n": len(suf_pool), "sufficiency_flip": paired_rate(suf_flip),
            "sufficiency_k": sum(suf_flip),
            "placebo_n": len(pla_flip), "placebo_flip": paired_rate(pla_flip),
            "placebo_k": sum(pla_flip),
        })
        r = results[op]
        print(f"    necessity  : block on correct-and-used     "
              f"{r['necessity_flip']:>6.1%} flip wrong  (n={r['necessity_n']})")
        print(f"    sufficiency: force on wrong-and-unused     "
              f"{r['sufficiency_flip']:>6.1%} flip right (n={r['sufficiency_n']})")
        print(f"    placebo    : force on correct-no-condition "
              f"{r['placebo_flip']:>6.1%} flip wrong  (n={r['placebo_n']})")

    # seed-tagged evidence: the acceptance aggregator enforces one summary
    # per declared seed — a single global file cannot prove seed coverage
    m_seed = re.search(r"_seed(\d+)$", args.run.rstrip("/"))
    tag = f"_seed{m_seed.group(1)}" if m_seed else ""
    out_path = os.path.join(args.out, f"interventions_summary{tag}.json")
    with open(out_path, "w") as fh:
        json.dump({"run": args.run, "results": results}, fh, indent=2)
    print(f"\nWritten to {out_path}")

    gate = Gate("Operator causal interventions (R8 usage -> R9 do())")
    for op, r in results.items():
        ok, d = rate_at_least(r["used_pos"], r["n_cond"],
                              point_min=0.15, ci_floor=0.10)
        gate.check(f"{op} R8: used when condition present (>=15%)", ok,
                   d + (" — " + r["verdict"] if not ok else ""))
        ok, d = rate_at_most(r["used_neg"], r["n_ctrl"],
                             point_max=0.05, ci_ceiling=0.15)
        gate.check(f"{op} R8: rarely used without condition (<=5%)", ok, d)
        if "necessity_k" not in r:
            continue
        # replay is deterministic, so every necessity flip IS caused by the
        # do() — the statistics quantify magnitude, not existence
        ok, d = rate_at_least(r["necessity_k"], r["necessity_n"],
                              point_min=0.10, ci_floor=0.05)
        gate.check(f"{op} R9: do(no-{op}) flips >=10%", ok,
                   d + " — below this, the operator decorated answers it "
                   "did not cause")
        ok, d = rate_at_most(r["placebo_k"], r["placebo_n"],
                             point_max=0.15, ci_ceiling=0.30)
        gate.check(f"{op} R9: placebo damage <=15%", ok,
                   d + " — forcing where the condition is absent should be "
                   "near-harmless")
        if r["necessity_n"] >= 20 and r["placebo_n"] >= 20:
            t = two_prop_test(r["necessity_k"], r["necessity_n"],
                              r["placebo_k"], r["placebo_n"])
            print(f"  necessity-vs-placebo contrast for {op}: "
                  f"Δ={t['d']:+.1%} [{t['lo']:+.1%},{t['hi']:+.1%}], "
                  f"z={t['z']:.2f}, p={t['p']:.4f}")
    ok = gate.report()
    print("  Verdict vocabulary: PASS | FAIL | NOT-TESTABLE | USAGE-FAILURE "
          "(see per-operator lines above; do not lump them scientifically).\n"
          "  Sufficiency rates are reported with CIs, not gated: forcing one "
          "action at t=T//2 is a crude do().")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
