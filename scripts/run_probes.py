"""Representation probes (R0/R1/R4). Order is the method:

R0  probe self-validation on PLANTED synthetic encodings (>95% recovered,
    shuffled labels ~ chance). If R0 fails the instrument is broken and NO
    claim about the model is reported.
R1  location probes on symbreak pairs — bound slot (linear + nonlinear),
    wrong slot, pooled Z, encoder H at the variable's anchor — plus the
    Case A/B/C/D diagnosis of where the value actually lives.
R4  strict binding gates (--strict): bound >= chance + 20pts and
    bound - wrong_slot >= 15pts with exact McNemar. Default gates are the
    softer pre-binding thresholds (10pts) for standard-substrate runs.

Run on a trained FAR² checkpoint; --control gives the fixed-slot side-by-side
the "concept separation vs slot-count increase" claim needs."""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from far2.eval.metrics import Gate
from far2.eval.probes import pair_value_probe, probe_selfcheck
from far2.eval.stats import ALPHA, rate_above_baseline
from far2.train import Trainer
from far2.utils import pick_device


def probe_run(run_dir: str, device, args):
    model, _ = Trainer.load_model(os.path.join(run_dir, "ckpt.pt"), device)
    model.eval()
    if hasattr(model, "set_unlocked"):
        model.set_unlocked({"expand", "split", "halt", "reconsider", "epistemic"})
    return pair_value_probe(model, device, n=args.n, difficulty=args.difficulty,
                            T=args.iters, seed=args.seed, family=args.family)


def show(tag: str, r: dict) -> None:
    if "error" in r:
        print(f"  {tag:<10} NOT-TESTABLE: {r['error']} (n={r['n_episodes']})")
        return
    print(f"  {tag:<10} bound {r['bound']:.1%} (mlp {r['bound_mlp']:.1%}) · "
          f"wrong-slot {r['wrong_slot']:.1%} · pooled {r['pooled']:.1%} · "
          f"H-anchor {r['H_anchor']:.1%} · chance {r['chance']:.1%} · "
          f"separation {r['separation_index']:+.1%} (n={r['n_episodes']} ep)")
    print(f"  {'':<10} R1 diagnosis: {r['diagnosis']}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="R0 selfcheck + R1 location probes"
                                            " + R4 binding gates")
    p.add_argument("--run", default="runs/phase3b_adaptive_bind_seed0")
    p.add_argument("--control", default=None,
                   help="fixed-slot control run dir (e.g. phase3a_fixed16)")
    p.add_argument("--family", default="symbreak",
                   help="symbreak, or sym_poly for the transfer-side probe")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs")
    p.add_argument("--strict", action="store_true",
                   help="R4 thresholds: bound>=chance+20pts, separation>=15pts")
    args = p.parse_args()

    sc = probe_selfcheck()
    print(f"[R0] planted {sc['planted']:.1%} (need >95%) · shuffled "
          f"{sc['shuffled']:.1%} (chance {sc['chance']:.0%}) -> "
          f"{'ok' if sc['ok'] else 'BROKEN'}")
    if not sc["ok"]:
        print("[R0] the probe pipeline itself failed self-validation — "
              "refusing to report anything about the model")
        sys.exit(1)

    device = pick_device(args.device)
    print(f"[R1] {args.family} d{args.difficulty} · T={args.iters} · "
          f"n={args.n} rows · device={device}\n")
    results = {"selfcheck": sc, "run": probe_run(args.run, device, args)}
    show("adaptive", results["run"])
    if args.control:
        results["control"] = probe_run(args.control, device, args)
        show("control", results["control"])

    # seed-tagged evidence: the acceptance aggregator enforces one summary
    # per declared seed — a single global file cannot prove seed coverage
    m_seed = re.search(r"_seed(\d+)$", args.run.rstrip("/"))
    stag = f"_seed{m_seed.group(1)}" if m_seed else ""
    out_path = os.path.join(args.out, f"probes_summary{stag}.json")
    with open(out_path, "w") as fh:
        json.dump({"run": args.run, "control": args.control,
                   "strict": args.strict, **results}, fh, indent=2)
    print(f"\nWritten to {out_path}")

    r = results["run"]
    m_bound, m_sep = (0.20, 0.15) if args.strict else (0.10, 0.10)
    tier = "R4 strict" if args.strict else "R1 baseline"
    gate = Gate(f"Representation probes ({tier})")
    if "error" in r:
        gate.check("probe testable", False,
                   f"NOT-TESTABLE — {r['error']}; this is a population "
                   "problem, not a model verdict")
    else:
        ok, d = rate_above_baseline(r["bound_k"], r["bound_n"],
                                    p0=r["chance"], margin=m_bound)
        gate.check(f"bound-slot decode >= chance + {m_bound:.0%} (exact "
                   "binomial)", ok, d)
        mc_ok = (r["separation_index"] >= m_sep
                 and r["mcnemar_p"] == r["mcnemar_p"]
                 and r["mcnemar_p"] < ALPHA)
        gate.check(f"bound - wrong_slot >= {m_sep:.0%} AND exact McNemar "
                   "p<0.05", mc_ok,
                   f"separation {r['separation_index']:+.1%}, "
                   f"b={r['mcnemar_b']}/c={r['mcnemar_c']}, "
                   f"p={r['mcnemar_p']:.2e} — if not, the pair shares one "
                   "representation and SPLIT claims must be reported as "
                   "capacity allocation, not concept separation")
    sys.exit(0 if gate.report() else 1)


if __name__ == "__main__":
    main()
