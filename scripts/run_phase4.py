"""Phase 4 — representation-insufficiency signal tournament, run BEFORE any
free controller training (P3b consumes the winner; no circularity). Scores
every candidate signal by AUROC against the requires_distinction condition on
the Phase 3a oracle model, and writes the selection to runs/selected_signal.json.

BLOCKED evaluation: the checkpoint was trained WITH oracle splitting, so a
straight eval would let post-action state leak into the tournament (on an
oracle model the forced split IS the condition — a detector scored on the
acted-on state reads the effect, not the cause). Actions are do()-blocked for
the whole episode: every candidate, including the final-state binding overlap,
is measured on pre-action state only."""

import json
import os

from _common import apply_mini, finish, parser

from far2.eval import EvalSpec, evaluate
from far2.eval.metrics import Gate, signal_aurocs
from far2.train import Trainer
from far2.utils import pick_device


def main():
    p = parser("Phase 4: collision-signal AUROC tournament", steps=0,
               eval_samples=400)
    p.add_argument("--run", default=None,
                   help="phase 3a oracle-split run dir "
                        "(default runs/phase3a_oracle_split[_bind]_seed<seed>)")
    p.add_argument("--substrate", choices=["standard", "bind"],
                   default="standard")
    args = p.parse_args()
    apply_mini(args)
    tagged = "" if args.substrate == "standard" else "_bind"
    run_dir = args.run or os.path.join(
        args.out, f"phase3a_oracle_split{tagged}_seed{args.seed}")
    device = pick_device(args.device)
    model, _ = Trainer.load_model(os.path.join(run_dir, "ckpt.pt"), device)
    model.set_unlocked({"expand", "split"})

    recs = evaluate(model, [EvalSpec("symbreak", [1, 2, 4, 6],
                                     n_samples=args.eval_samples)],
                    device, out_path=os.path.join(run_dir, "eval", "phase4.jsonl"),
                    intervene={"block": ("expand", "split")})
    aurocs = signal_aurocs(recs)

    print("\nsignal AUROC vs requires_distinction — BLOCKED evaluation "
          "(expand/split do()-blocked all episode: every candidate is "
          "pre-action state):")
    ranked = sorted(aurocs.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1))
    for k, v in ranked:
        marker = ("  <- C = Sim x Div (the FAR2 hypothesis)" if k == "C" else
                  "  <- searched over all pairs (runtime-legitimate)"
                  if k == "bind" else
                  "  (needs ground truth / post-action: displayed, NOT "
                  "selectable)" if k.endswith(("_post", "_probe")) else "")
        print(f"  {k:<18} {v:.3f}{marker}")

    eligible = [(k, v) for k, v in ranked
                if not k.endswith(("_post", "_probe"))]
    best_name, best = eligible[0] if eligible else (None, float("nan"))
    mapped = {"C_max_over_iters": "C"}

    gate = Gate("Phase 4 · signal tournament")
    sel_ok = best == best and best > 0.55
    gate.check("best signal beats chance", sel_ok,
               f"{best_name} AUROC {best:.3f}")
    c = aurocs.get("C", float("nan"))
    gate.check("C (Sim x Div) reported against components", c == c,
               f"C {c:.3f} vs sim {aurocs.get('sim', float('nan')):.3f} / "
               f"jsd {aurocs.get('jsd', float('nan')):.3f} — winner: {best_name}")

    # a FAILED tournament must never persist a selection: phase3b would
    # silently train on a signal that does not detect the condition
    sel_path = os.path.join(args.out, "selected_signal.json")
    if sel_ok:
        selection = {"signal": mapped.get(best_name, best_name),
                     "aurocs": aurocs, "substrate": args.substrate,
                     "source_run": run_dir}
        with open(sel_path, "w") as fh:
            json.dump(selection, fh, indent=2)
        print(f"\n  selection written to {sel_path} — run_phase3b consumes it.")
    else:
        if os.path.exists(sel_path):
            os.remove(sel_path)
            print(f"\n  removed stale {sel_path}")
        print("  NO selection written (gate failed) — phase3b will fall back "
              "to the substrate default signal.")
    finish(gate.report())


if __name__ == "__main__":
    main()
