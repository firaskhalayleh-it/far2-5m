"""Wall-clock estimate for scripts/run_foundation_curve.py without running it.

Times a handful of real optimizer steps per (arm, curriculum stage) through the
unmodified Trainer.train() loop, and one small evaluation pass per arm through
the unmodified evaluate(), then scales to the 4,000-step / 400-sample contract.
Nothing is written into the repository; outputs go to a temp directory.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import torch  # noqa: E402

import run_foundation_curve as fc  # noqa: E402
from far2.config import TrainConfig  # noqa: E402
from far2.eval import evaluate  # noqa: E402
from far2.models import build_model  # noqa: E402
from far2.presets import get_preset  # noqa: E402
from far2.train import Trainer  # noqa: E402
from far2.utils import pick_device, set_seed  # noqa: E402

STEPS, EVAL_N, CHECKPOINTS = 4000, 400, 5


def arms():
    base = replace(get_preset("far2_bind"), n_iters_train=8, init_slots=14)
    return {
        "far2_fixed14_warm": (replace(base, workspace_init="input"), 0.0, "final"),
        "far2_fixed14_seed": (replace(base, workspace_init="learned"), 0.0, "final"),
        "far2_fixed14_warm_all": (replace(base, workspace_init="input"), 0.5, "all"),
        "slot_recurrent_bind": (replace(get_preset("slot_recurrent_bind"),
                                        workspace_init="input"), 0.0, "final"),
        "transformer_5m": (get_preset("transformer_5m"), 0.0, "final"),
        "transformer_bind_match": (get_preset("transformer_bind_match"), 0.0, "final"),
    }


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def time_steps(cfg, alpha, supervision, stage, device_name, n, warm, tmp):
    """Seconds per optimizer step for one stage, via the real train() loop."""
    device = pick_device(device_name)

    tc = TrainConfig(steps=warm + n + 1, batch_size=128, device=device_name, seed=0,
                     out_dir=tempfile.mkdtemp(dir=tmp), warmup=50, ema=0.999,
                     ipt_alpha=alpha, cycle_supervision=supervision,
                     log_every=10 ** 9)
    set_seed(0)
    tr = Trainer(build_model(cfg), cfg, tc, [replace(stage, frac=1.0)])
    tr.save = lambda: None  # final checkpoint write is not part of a step
    stamps, inner = [], tr.opt.step

    def stamped(*a, **k):  # one stamp per full loop cycle: data, fwd, bwd, clip, step, EMA
        out = inner(*a, **k)
        sync(device)
        stamps.append(time.perf_counter())
        return out

    tr.opt.step = stamped
    tr.train()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])][warm:]
    return sum(gaps) / len(gaps)


def time_eval(cfg, device_name, n, tmp):
    """Seconds for one checkpoint's evaluation at n samples per cell."""
    device = pick_device(device_name)
    model = build_model(cfg).to(device)
    if hasattr(model, "set_unlocked"):
        model.set_unlocked(set())
    model.eval()
    recurrent = cfg.kind == "far2"
    evaluate(model, fc._specs(2, recurrent), device, seed=1)  # compile kernels first
    sync(device)
    t0 = time.perf_counter()
    recs = evaluate(model, fc._specs(n, recurrent), device, seed=1234,
                    out_path=os.path.join(tempfile.mkdtemp(dir=tmp), "e.jsonl"))
    sync(device)
    return time.perf_counter() - t0, len(recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--n", type=int, default=6, help="timed steps per cell")
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--eval-n", type=int, default=32)
    p.add_argument("--arms", default="")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    bounds = [int(f * STEPS) for f in (0.2, 0.5, 1.0)]
    per_stage = [bounds[0], bounds[1] - bounds[0], bounds[2] - bounds[1]]
    chosen = arms()
    if args.arms:
        chosen = {k: chosen[k] for k in args.arms.split(",")}
    print(f"torch {torch.__version__} device={args.device} "
          f"threads={torch.get_num_threads()}", flush=True)
    report, total = {}, 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for name, (cfg, alpha, sup) in chosen.items():
            stages = fc._stages(alpha)
            secs = [time_steps(cfg, alpha, sup, s, args.device, args.n, args.warm, tmp)
                    for s in stages]
            train_s = sum(n * s for n, s in zip(per_stage, secs))
            ev_s, ev_n = time_eval(cfg, args.device, args.eval_n, tmp)
            eval_s = ev_s * (EVAL_N / args.eval_n) * CHECKPOINTS
            if name == "far2_fixed14_warm":  # + initialization replay, 3 modes x 3 cells @T8
                eval_s += ev_s / ev_n * 3 * 3 * EVAL_N
            report[name] = {"s_per_step_by_stage": [round(x, 3) for x in secs],
                            "train_min": round(train_s / 60, 1),
                            "eval_min": round(eval_s / 60, 1)}
            total += train_s + eval_s
            print(f"{name:26s} step s/stage={report[name]['s_per_step_by_stage']} "
                  f"train={train_s / 60:6.1f} min  eval={eval_s / 60:5.1f} min", flush=True)
    print(f"TOTAL ({len(chosen)} arms): {total / 3600:.2f} h", flush=True)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
