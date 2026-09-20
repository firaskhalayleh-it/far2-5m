"""Two-minute end-to-end proof: all five model kinds train their shared core
on tiny configs, loss falls, each learns a real exact-mathematics micro-task
above the modal baseline, the eval harness runs, and the report renders.

Optional FAR2 operators stay neutral here: mechanism-specific training and
execution belong to run_campaign.py --mini. This smoke is a core-retention
and wiring check, not a scientific phase gate. Run: python scripts/smoke_test.py
"""

import _common  # noqa: F401  (sys.path setup)
import os
import sys
from collections import Counter

from far2.config import TrainConfig
from far2.data.registry import Mix
from far2.eval import EvalSpec, evaluate
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer
from far2.utils import set_seed
from far2.utils.logging import read_jsonl
from representation_preflight import run_preflight
from learned_layer_preflight import run_learned_preflight

OUT = os.environ.get("FAR2_SMOKE_OUT", "runs/smoke")
# Shallow carry data exercises variable-length arithmetic while a heavier
# one-digit multiplication stream gives every architecture a task it can
# learn exactly at smoke scale. Multi-digit competence belongs to P0/pilot.
MIXES = [Mix("carry_add", 0.5, 0, 1), Mix("multiply", 1.5, 1, 1)]
MIN_EXACT_ACC = 0.20
MIN_MODAL_GAIN = 0.10


def smoke_one(preset: str) -> float:
    mc = get_preset(preset)
    tc = TrainConfig(steps=600, batch_size=32, lr=1e-3, warmup=20, device="cpu",
                     out_dir=os.path.join(OUT, preset), log_every=25, ema=0.99,
                     seed=0, ipt_alpha=0.0, cycle_supervision="final")
    stage = Stage("smoke", 1.0, MIXES, depth="fixed")
    # Initialization is part of the declared seed, matching the campaign
    # helpers and making the competence gate reproducible.
    set_seed(tc.seed)
    model = build_model(mc)
    trainer = Trainer(model, mc, tc, [stage])
    trainer.train()

    log = read_jsonl(os.path.join(tc.out_dir, "train_log.jsonl"))
    first, last = log[0]["loss"], log[-1]["loss"]
    drop = (first - last) / first
    assert drop > 0.3, f"{preset}: loss barely moved ({first:.3f} -> {last:.3f})"

    with trainer.eval_weights():
        recs = evaluate(model, [EvalSpec("multiply", [1], n_samples=96)],
                        trainer.device, out_path=os.path.join(tc.out_dir, "eval", "smoke.jsonl"))
    acc = sum(r["correct"] for r in recs) / len(recs)
    token_acc = (sum(sum(a == b for a, b in zip(r["pred"], r["target"]))
                     for r in recs)
                 / sum(len(r["target"]) for r in recs))
    modal = max(Counter(tuple(r["target"]) for r in recs).values()) / len(recs)
    assert acc >= MIN_EXACT_ACC and acc >= modal + MIN_MODAL_GAIN, (
        f"{preset}: loss fell but exact math was not learned "
        f"(acc={acc:.1%}, modal={modal:.1%})")
    print(f"  {preset}: loss {first:.3f} -> {last:.3f} (-{drop:.0%}), "
          f"multiply@1 exact {acc:.0%} (modal {modal:.0%}), "
          f"token {token_acc:.0%}, records schema ok "
          f"({sorted(recs[0].keys())[:6]}...)")
    return acc


def main():
    # Mechanical representation laws are prerequisites.  Training must never
    # hide a broken layer behind an aggregate smoke score.
    run_preflight()
    learned = run_learned_preflight(
        steps=300, batch_size=64, device="auto", seed=451,
        out_dir=os.path.join(OUT, "learned_preflight"))
    assert learned["passed"], "learned-layer foundation preflight failed"
    smoke_one("transformer_tiny")
    smoke_one("recurrent_tiny")
    core = smoke_one("far2_tiny")
    full = smoke_one("far2_tiny_full")
    bind = smoke_one("far2_tiny_bind")
    for name, acc in (("far2_tiny_full", full), ("far2_tiny_bind", bind)):
        assert acc >= core - 0.10, (
            f"{name}: optional architecture does not preserve core competence "
            f"({acc:.1%} vs far2_tiny {core:.1%})")

    import subprocess
    r = subprocess.run([sys.executable, "scripts/make_report.py", OUT,
                        "--out", os.path.join(OUT, "report.html")],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"report failed: {r.stderr[-800:]}"
    print(f"  report rendered: {OUT}/report.html")
    print("\nSMOKE TEST PASSED — the pipeline is real.")


if __name__ == "__main__":
    main()
