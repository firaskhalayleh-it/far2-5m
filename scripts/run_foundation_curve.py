"""v0.4.52 capability-first foundation curve on the real P3 mixture.

Fresh in-distribution competence (symbreak R<=4) is kept separate from OOD
representation load (R=6), and recurrent models are read at Z0/T1/T4/T8.
Every arm receives the same staged examples, batch, seed, and checkpoints.
The selected FAR2 arm is answer-only and operator-free; learned-seed and
all-cycle arms are declared optimization diagnostics, not hidden scaffolding.
"""

import _common  # noqa: F401
import argparse
import json
import os
from dataclasses import replace

import torch

from far2.config import TrainConfig
from far2.eval import EvalSpec, evaluate
from far2.eval.metrics import accuracy, token_accuracy
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer, core_mixes, fam
from far2.utils import pick_device, set_seed


VERSION = "0.4.52"
ID_SYMBREAK = (1, 2, 4)       # training frontier ends at R=4
OOD_SYMBREAK = (6,)            # reported separately; never pooled into readiness
ITERATION_READOUTS = (0, 1, 4, 8)
READINESS_EXACT = 0.20
READINESS_TOKEN = 0.75
RECURRENCE_MARGIN = 0.05
MIN_CAPABILITY_STEPS = 4000
MIN_CAPABILITY_EVAL = 400


def _mix():
    return core_mixes(0.4) + [fam("linsys", 0.6), fam("symbreak", 1.5)]


def _stages(ipt_alpha):
    """Learn easy plumbing first, then rules, then the real P3 pressure.

    No stage unlocks an operator or supplies oracle/entity/binding labels.
    The only non-final answer supervision is confined to the named all-cycle
    diagnostic arm.
    """
    return [
        Stage("foundation_easy", 0.20, core_mixes(1.0, hard=False),
              depth="fixed", ipt_alpha=ipt_alpha),
        Stage("foundation_rules", 0.30,
              core_mixes(0.6, hard=True) + [fam("linsys", 0.8, easy=True)],
              depth="fixed", ipt_alpha=ipt_alpha),
        Stage("foundation_real", 0.50, _mix(), depth="fixed",
              ipt_alpha=ipt_alpha, bind_weight=0.0),
    ]


def _specs(n, recurrent):
    iters = list(ITERATION_READOUTS) if recurrent else [None]
    return [EvalSpec("symbreak", list(ID_SYMBREAK), iters, n_samples=n),
            EvalSpec("symbreak", list(OOD_SYMBREAK), iters, n_samples=n),
            EvalSpec("linsys", [2, 3, 4], iters, n_samples=n),
            EvalSpec("linsys", [6], iters, n_samples=n)]


def _rows(recs, family, difficulties, n_iters=None):
    return [r for r in recs if r["family"] == family
            and r["difficulty"] in difficulties
            and (n_iters is None or r["n_iters"] == n_iters)]


def _score(recs):
    return {"exact": accuracy(recs), "token": token_accuracy(recs),
            "n": len(recs)}


def _capability(recs, recurrent):
    if not recurrent:
        ident = _score(_rows(recs, "symbreak", ID_SYMBREAK))
        return {"readouts": {"fixed_depth": {
                    "id_symbreak": ident,
                    "ood_symbreak": _score(_rows(
                        recs, "symbreak", OOD_SYMBREAK)),
                    "id_linsys": _score(_rows(recs, "linsys", (2, 3, 4))),
                    "ood_linsys": _score(_rows(recs, "linsys", (6,))),
                }},
                "fresh_id_ready": (ident["exact"] >= READINESS_EXACT
                                   and ident["token"] >= READINESS_TOKEN)}

    curve = {}
    for T in ITERATION_READOUTS:
        curve[str(T)] = {
            "id_symbreak": _score(_rows(recs, "symbreak", ID_SYMBREAK, T)),
            "ood_symbreak": _score(_rows(recs, "symbreak", OOD_SYMBREAK, T)),
            "id_linsys": _score(_rows(recs, "linsys", (2, 3, 4), T)),
            "ood_linsys": _score(_rows(recs, "linsys", (6,), T)),
        }
    z0, t1, t8 = (curve[str(t)]["id_symbreak"] for t in (0, 1, 8))
    return {"readouts": curve,
            "fresh_id_ready": (t8["exact"] >= READINESS_EXACT
                               and t8["token"] >= READINESS_TOKEN),
            "z0_to_t8_exact": t8["exact"] - z0["exact"],
            "t1_to_t8_exact": t8["exact"] - t1["exact"],
            "t1_to_t8_token": t8["token"] - t1["token"]}


def _initialization_replay(model, cfg, device, n, run_dir):
    """Hold EMA weights fixed and change only the workspace initializer."""
    state = model.state_dict()
    out = {}
    for mode in ("input", "learned", "input_shuffled"):
        variant = build_model(replace(cfg, workspace_init=mode)).to(device)
        variant.load_state_dict(state)
        variant.set_unlocked(set())
        recs = evaluate(
            variant,
            [EvalSpec("symbreak", list(ID_SYMBREAK), [8], n_samples=n)],
            device, seed=1234,
            out_path=os.path.join(run_dir, "eval", f"init_{mode}_T8.jsonl"))
        out[mode] = _score(recs)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--checkpoints", default="250,500,1000,2000,4000")
    p.add_argument("--eval-samples", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="default: runs_v0452/foundation_curve_seed<seed>")
    p.add_argument("--arms", default="",
                   help="comma-separated arm names; empty runs every declared arm")
    args = p.parse_args()
    if args.out is None:
        args.out = f"runs_v0452/foundation_curve_seed{args.seed}"
    points = tuple(sorted({int(x) for x in args.checkpoints.split(",")
                           if int(x) <= args.steps} | {args.steps}))
    # Preserve every FAR2-bind module for the parameter-identity contract.
    # The empty stage unlock set makes all optional mechanisms behaviorally
    # inert without deleting their parameters.
    base = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=14)
    arms = {
        "far2_fixed14_warm": (replace(base, workspace_init="input"), 0.0, "final"),
        "far2_fixed14_seed": (replace(base, workspace_init="learned"), 0.0, "final"),
        "far2_fixed14_warm_all": (replace(base, workspace_init="input"), 0.5, "all"),
        "slot_recurrent_bind": (replace(get_preset("slot_recurrent_bind"),
                                         workspace_init="input"), 0.0, "final"),
        "transformer_5m": (get_preset("transformer_5m"), 0.0, "final"),
        "transformer_bind_match": (get_preset("transformer_bind_match"), 0.0, "final"),
    }
    if args.arms:
        requested = [x.strip() for x in args.arms.split(",") if x.strip()]
        unknown = sorted(set(requested) - set(arms))
        if unknown:
            p.error("unknown arms: " + ", ".join(unknown))
        arms = {name: arms[name] for name in requested}
    device = pick_device(args.device)
    summary = {"version": VERSION, "contract": {
        "steps": args.steps, "checkpoints": list(points),
        "eval_samples_per_cell": args.eval_samples, "batch": args.batch,
        "seed": args.seed, "fresh_eval_seed": 1234,
        "id_symbreak": list(ID_SYMBREAK),
        "ood_symbreak": list(OOD_SYMBREAK),
        "iteration_readouts": list(ITERATION_READOUTS),
        "readiness": {"exact": READINESS_EXACT, "token": READINESS_TOKEN},
        "recurrence_margin": RECURRENCE_MARGIN,
        "bind_weight": 0.0, "oracle_weight": 0.0,
        "operator_unlocks": [], "no_transfer_training": True,
    }, "arm_contracts": {}, "arms": {}, "initialization_replays": {}}
    for name, (cfg, ipt_alpha, supervision) in arms.items():
        run_dir = os.path.join(args.out, name)
        tc = TrainConfig(
            steps=args.steps, batch_size=args.batch, device=args.device,
            seed=args.seed, out_dir=run_dir, warmup=max(50, args.steps // 15),
            eval_samples=args.eval_samples, ema=0.999,
            ipt_alpha=ipt_alpha, cycle_supervision=supervision,
            checkpoint_steps=points)
        summary["arm_contracts"][name] = {
            "cycle_supervision": supervision,
            "intermediate_loss_weight": ipt_alpha,
            "bind_weight": 0.0, "oracle_weight": 0.0,
            "operator_unlocks": [],
        }
        set_seed(args.seed)
        trainer = Trainer(build_model(cfg), cfg, tc, _stages(ipt_alpha))
        print(f"[{name}] params={trainer.model.param_count():,} "
              f"device={device} steps={args.steps} supervision={supervision} "
              f"alpha={ipt_alpha}", flush=True)
        trainer.train()
        arm_rows = []
        for point in points:
            path = os.path.join(run_dir, f"ckpt_step{point}.pt")
            model, loaded_cfg = Trainer.load_model(path, device=device)
            recurrent = loaded_cfg.kind == "far2"
            recs = evaluate(model, _specs(args.eval_samples, recurrent), device,
                            seed=1234, out_path=os.path.join(
                                run_dir, "eval", f"step{point}.jsonl"))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            capability = _capability(recs, recurrent)
            row = {
                "step": point, "params": model.param_count(),
                "capability": capability,
                **ckpt.get("train_state", {}),
            }
            arm_rows.append(row)
            if name == "far2_fixed14_warm" and point == args.steps:
                summary["initialization_replays"][name] = _initialization_replay(
                    model, loaded_cfg, device, args.eval_samples, run_dir)
            if recurrent:
                id8 = capability["readouts"]["8"]["id_symbreak"]
                print(f"  step {point}: ID exact={id8['exact']:.1%} "
                      f"token={id8['token']:.1%} "
                      f"Z0->T8={capability['z0_to_t8_exact']:+.1%} "
                      f"T1->T8={capability['t1_to_t8_exact']:+.1%} "
                      f"train_FLOPs={row.get('train_flops', 0):,.0f}", flush=True)
            else:
                ident = capability["readouts"]["fixed_depth"]["id_symbreak"]
                print(f"  step {point}: ID exact={ident['exact']:.1%} "
                      f"token={ident['token']:.1%} "
                      f"train_FLOPs={row.get('train_flops', 0):,.0f}", flush=True)
        summary["arms"][name] = arm_rows
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "foundation_curve.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    required = {"far2_fixed14_warm", "far2_fixed14_seed",
                "far2_fixed14_warm_all", "slot_recurrent_bind",
                "transformer_bind_match"}
    missing = sorted(required - set(summary["arms"]))
    if missing:
        verdict = {"verdict": "NOT-TESTABLE", "missing_arms": missing}
    elif (args.steps < MIN_CAPABILITY_STEPS
          or args.eval_samples < MIN_CAPABILITY_EVAL):
        verdict = {"verdict": "NOT-TESTABLE",
                   "reason": "smoke/mini budgets cannot pass capability",
                   "required_steps": MIN_CAPABILITY_STEPS,
                   "required_eval_samples": MIN_CAPABILITY_EVAL}
    else:
        warm = summary["arms"]["far2_fixed14_warm"][-1]["capability"]
        checks = {
            "fresh_id_competence": warm["fresh_id_ready"],
            "recurrent_gain_over_z0": (
                warm["z0_to_t8_exact"] >= RECURRENCE_MARGIN),
            "recurrent_gain_after_t1": (
                warm["t1_to_t8_exact"] >= RECURRENCE_MARGIN),
            "input_conditioning_is_causal": (
                summary["initialization_replays"]["far2_fixed14_warm"]
                ["input"]["exact"]
                >= summary["initialization_replays"]["far2_fixed14_warm"]
                ["input_shuffled"]["exact"] + RECURRENCE_MARGIN),
            "selected_arm_is_answer_only": (
                summary["arm_contracts"]["far2_fixed14_warm"]
                == {"cycle_supervision": "final",
                    "intermediate_loss_weight": 0.0,
                    "bind_weight": 0.0, "oracle_weight": 0.0,
                    "operator_unlocks": []}),
            "matched_controls_complete": True,
        }
        verdict = {"verdict": ("PASS" if all(checks.values())
                               else "NOT-READY"),
                   "checks": checks,
                   "note": "OOD cells are separate; Phase 2 and transfer own those claims"}
    summary["capability_verdict"] = verdict
    with open(os.path.join(args.out, "foundation_curve.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n[v0.4.52 capability] " + json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
