"""Final phase — CROSS-DOMAIN MATHEMATICAL TRANSFER on the held-out hidden
benchmark. The headline question of the 5M program:

  SPLIT       trained on linear symmetry breaking      -> sym_poly
              (symmetric POLYNOMIAL constraints, unseen grammar)
  RECONSIDER  trained on extraneous sqrt roots         -> rational_trap
              (excluded-value trap, unseen grammar)
  recurrence  trained on arithmetic/expression depth   -> iter_apply
              (k-fold function application, unseen family)
  epistemics  trained on linear-system status          -> congruence
              (modular underdetermination, unseen grammar)

No family token (shuffled everywhere), no oracle, no retraining. If FAR²
detects the CONDITION — splits with binding separation on sym_poly, discards
the tempting excluded root, abstains on modular ambiguity — where the
slot-recurrent control cannot, it has learned reusable operations about
cognition rather than 'linear systems require SPLIT'. See docs/ROADMAP.md."""

import hashlib
import json
import os

from _common import apply_mini, finish, parser

from far2.data.registry import TRANSFER_GENERATORS
from far2.eval import EvalSpec, evaluate
from far2.eval.metrics import (Gate, acc_by, accuracy, assertion_counts,
                               assertion_rates, binding_separation,
                               recovery_counts, recovery_rate,
                               split_selectivity)
from far2.eval.stats import rate_at_most, superiority
from far2.train import Trainer
from far2.utils import pick_device


def _training_families(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        raise RuntimeError(f"transfer requires the serialized training contract: {path}")
    with open(path) as fh:
        cfg = json.load(fh)
    return sorted({m["family"] for stage in cfg.get("curriculum", [])
                   for m in stage.get("mixes", [])})


def _state_fingerprint(model):
    """Prove evaluation did not mutate any learned tensor."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def specs(n):
    out = []
    for f, g in TRANSFER_GENERATORS.items():
        mid = (g.min_difficulty + g.max_difficulty) // 2
        out.append(EvalSpec(f, sorted({g.min_difficulty, mid,
                                       g.max_difficulty}), n_samples=n))
    return out


def main():
    p = parser("Transfer: unseen-family mathematical mechanism test", steps=0,
               eval_samples=300)
    p.add_argument("--ckpt", default=None,
                   help="trained FAR² run dir (default runs/ablate_far2_bind_seed<seed>, "
                        "falling back to ablate_far2_full)")
    p.add_argument("--control-ckpt", default=None,
                   help="slot-recurrent control run dir")
    args = p.parse_args()
    apply_mini(args)
    device = pick_device(args.device)

    ckpt = args.ckpt
    if ckpt is None:
        for cand in (f"ablate_far2_bind_seed{args.seed}",
                     f"ablate_far2_full_seed{args.seed}",
                     f"phase3b_adaptive_bind_seed{args.seed}",
                     f"phase3b_adaptive_seed{args.seed}"):
            if os.path.exists(os.path.join(args.out, cand, "ckpt.pt")):
                ckpt = os.path.join(args.out, cand)
                break
    assert ckpt, "no trained FAR² checkpoint found — run ablations/phase3b first"
    model, mc = Trainer.load_model(os.path.join(ckpt, "ckpt.pt"), device)
    trained_families = _training_families(ckpt)
    leaked = sorted(set(trained_families) & set(TRANSFER_GENERATORS))
    if leaked:
        raise RuntimeError("held-out transfer families appeared in training: "
                           + ", ".join(leaked))
    model.set_unlocked({"expand", "split", "halt", "reconsider", "epistemic"})
    # a transfer claim about an operator the checkpoint DOES NOT HAVE is not a
    # failed claim — it is NOT-TESTABLE from this checkpoint (train the
    # full-featured arm for those; see the deferred tier)
    caps = {"split": bool(mc.adaptive_slots), "reconsider": bool(mc.reconsider),
            "epistemic": bool(mc.epistemic)}
    print(f"[transfer] FAR² checkpoint: {ckpt}")
    print(f"[transfer] checkpoint operators: "
          + " · ".join(f"{k}={'on' if v else 'OFF'}" for k, v in caps.items()))
    far2_before = _state_fingerprint(model)
    far2_recs = evaluate(model, specs(args.eval_samples), device,
                         out_path=os.path.join(ckpt, "eval", "transfer.jsonl"))
    far2_after = _state_fingerprint(model)
    if far2_before != far2_after:
        raise RuntimeError("transfer evaluation mutated FAR2 parameters")

    ctrl_recs = None
    ctrl = args.control_ckpt
    if ctrl is None:
        for cand in (f"ablate_slot_recurrent_bind_seed{args.seed}",
                     f"ablate_slot_recurrent_5m_seed{args.seed}",
                     f"phase0_slot_recurrent_seed{args.seed}"):
            if os.path.exists(os.path.join(args.out, cand, "ckpt.pt")):
                ctrl = os.path.join(args.out, cand)
                break
    if ctrl and os.path.exists(os.path.join(ctrl, "ckpt.pt")):
        cmodel, _ = Trainer.load_model(os.path.join(ctrl, "ckpt.pt"), device)
        ctrl_families = _training_families(ctrl)
        leaked_ctrl = sorted(set(ctrl_families) & set(TRANSFER_GENERATORS))
        if leaked_ctrl:
            raise RuntimeError("control trained on held-out transfer families: "
                               + ", ".join(leaked_ctrl))
        print(f"[transfer] control checkpoint: {ctrl}")
        ctrl_before = _state_fingerprint(cmodel)
        ctrl_recs = evaluate(cmodel, specs(args.eval_samples), device,
                             out_path=os.path.join(ctrl, "eval", "transfer.jsonl"))
        if ctrl_before != _state_fingerprint(cmodel):
            raise RuntimeError("transfer evaluation mutated control parameters")
    else:
        print("[transfer] no slot-recurrent control found — the comparison "
              "gates will report NOT-TESTABLE")

    print("\naccuracy on HELD-OUT families:")
    print(f"  {'family':<16} {'FAR²':>8} {'slot-rec C':>11}")
    for f in TRANSFER_GENERATORS:
        a = accuracy(far2_recs, family=f)
        c = accuracy(ctrl_recs, family=f) if ctrl_recs else float("nan")
        print(f"  {f:<16} {a:>8.1%} {c:>11.1%}")

    coll = [r for r in far2_recs if r["family"] == "sym_poly"]
    sel = split_selectivity(coll)
    bind = binding_separation(coll)
    trap = [r for r in far2_recs if r["family"] == "rational_trap"]
    ctrl_trap = [r for r in (ctrl_recs or []) if r["family"] == "rational_trap"]
    rr, rr_ctrl = recovery_rate(trap), recovery_rate(ctrl_trap)
    unk = [r for r in far2_recs if r["family"] == "congruence"]
    unsupported, over = assertion_rates(unk)
    iter_curve = acc_by([r for r in far2_recs if r["family"] == "iter_apply"],
                        "difficulty")

    print(f"\n  spontaneous SPLIT on unseen sym_poly: when-required "
          f"{sel['rate_when_required']:.1%} vs control-queries "
          f"{sel['rate_on_control']:.1%}")
    print(f"  binding separation after split: "
          f"{bind.get('separated_after_split', float('nan')):.1%}")
    print(f"  excluded-root recovery (rational_trap): FAR² {rr:.1%} vs "
          f"slot-rec {rr_ctrl:.1%}")
    print(f"  unseen unknowables (congruence): unsupported {unsupported:.1%} "
          f"· over-abstention {over:.1%}")
    print("  iter_apply acc by depth: "
          + " · ".join(f"k={k} {v:.0%}" for k, v in iter_curve.items()))

    out_path = os.path.join(args.out, "transfer_summary.json")
    with open(out_path, "w") as fh:
        json.dump({"ckpt": ckpt, "control": ctrl,
                   "transfer_contract": {
                       "version": "0.4.52",
                       "evaluation_only": True,
                       "optimizer_updates_during_transfer": 0,
                       "parameters_unchanged": far2_before == far2_after,
                       "training_families": trained_families,
                       "held_out_training_overlap": leaked,
                       "checkpoint_state_sha256": far2_before,
                   },
                   "split_selectivity": sel, "binding": bind,
                   "recovery": rr, "recovery_control": rr_ctrl,
                   "unsupported": unsupported, "over_abstention": over,
                   "iter_apply_curve": iter_curve,
                   "acc": {f: accuracy(far2_recs, family=f)
                           for f in TRANSFER_GENERATORS}}, fh, indent=2)

    gate = Gate("Transfer · unseen-family mathematical mechanism evidence")
    if caps["split"]:
        ok, d = superiority(sel["tp"], sel["n_need"], sel["fp"], sel["n_ctrl"],
                            margin=0.10)
        gate.check("splits track the unseen condition (m=10pts, one-sided z)",
                   ok, d + " — no family token, no oracle")
    else:
        gate.check("splits track the unseen condition", False,
                   "NOT-TESTABLE: checkpoint has adaptive_slots off")
    if caps["reconsider"]:
        ok, d = superiority(*recovery_counts(trap), *recovery_counts(ctrl_trap),
                            margin=0.05)
        gate.check("trap recovery transfers: FAR² > slot-recurrent C (m=5pts)",
                   ok, d)
    else:
        gate.check("trap recovery transfers", False,
                   "NOT-TESTABLE: checkpoint lacks the RECONSIDER operator — "
                   "this claim needs the full-featured arm (deferred tier), "
                   "not a failed transfer")
    if caps["epistemic"]:
        (ku, nu), (ko, na) = assertion_counts(unk)
        ok_u, d_u = rate_at_most(ku, nu, point_max=0.20, ci_ceiling=0.35)
        ok_o, d_o = rate_at_most(ko, na, point_max=0.20, ci_ceiling=0.35)
        gate.check("unsupported assertions on unseen unknowables <=20%", ok_u, d_u)
        gate.check("over-abstention on unseen answerables <=20%", ok_o, d_o)
    else:
        gate.check("epistemic transfer (congruence)", False,
                   "NOT-TESTABLE: checkpoint lacks the epistemic head — "
                   "deferred-tier claim")
    print("\n  NOTE: this is the headline result of the 5M phase — report it "
          "with seed error bars either way.")
    finish(gate.report())


if __name__ == "__main__":
    main()
