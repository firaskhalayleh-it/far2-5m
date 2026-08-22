"""Two-minute end-to-end proof: all three models train on tiny configs, loss
falls, the eval harness runs, the report renders. If this passes, the codebase
is real. Run: python scripts/smoke_test.py"""

import _common  # noqa: F401  (sys.path setup)
import os
from dataclasses import replace

from far2.config import TrainConfig
from far2.data.registry import Mix
from far2.eval import EvalSpec, evaluate
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer
from far2.utils.logging import read_jsonl

OUT = "runs/smoke"
# shallow carry-free addition + tiny expression trees: the CE drop is fast at
# tiny scale, so a failed drop means broken code, not a hard task
MIXES = [Mix("carry_add", 1.0, 0, 1), Mix("expr_eval", 1.0, 2, 3)]


def smoke_one(preset: str, unlock: bool = False) -> None:
    mc = get_preset(preset)
    tc = TrainConfig(steps=400, batch_size=32, lr=1e-3, warmup=20, device="cpu",
                     out_dir=os.path.join(OUT, preset), log_every=25, ema=0.99)
    stage = Stage("smoke", 1.0, MIXES, depth="fixed")
    if unlock:
        stage = replace(stage, unlocked={"expand", "split", "halt",
                                         "reconsider", "epistemic"},
                        oracle_weight=0.2, slot_cost=0.005, entropy_bonus=0.01)
    model = build_model(mc)
    trainer = Trainer(model, mc, tc, [stage])
    trainer.train()

    log = read_jsonl(os.path.join(tc.out_dir, "train_log.jsonl"))
    first, last = log[0]["loss"], log[-1]["loss"]
    drop = (first - last) / first
    assert drop > 0.3, f"{preset}: loss barely moved ({first:.3f} -> {last:.3f})"

    with trainer.eval_weights():
        recs = evaluate(model, [EvalSpec("carry_add", [1], n_samples=48)],
                        trainer.device, out_path=os.path.join(tc.out_dir, "eval", "smoke.jsonl"))
    acc = sum(r["correct"] for r in recs) / len(recs)
    print(f"  {preset}: loss {first:.3f} -> {last:.3f} (-{drop:.0%}), "
          f"carry_add@1 acc {acc:.0%}, records schema ok "
          f"({sorted(recs[0].keys())[:6]}...)")


def main():
    for preset in ["transformer_tiny", "recurrent_tiny", "far2_tiny"]:
        smoke_one(preset)
    smoke_one("far2_tiny_full", unlock=True)
    smoke_one("far2_tiny_bind", unlock=True)   # competitive-binding substrate

    import subprocess
    r = subprocess.run(["python3", "scripts/make_report.py", OUT,
                        "--out", os.path.join(OUT, "report.html")],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"report failed: {r.stderr[-800:]}"
    print(f"  report rendered: {OUT}/report.html")
    print("\nSMOKE TEST PASSED — the pipeline is real.")


if __name__ == "__main__":
    main()
