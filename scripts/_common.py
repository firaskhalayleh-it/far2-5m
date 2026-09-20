"""Shared plumbing for phase scripts: arg parsing, train/eval wrappers.
Every script accepts --steps/--eval-samples/--device/--seed and --mini."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from far2.config import ModelConfig, TrainConfig
from far2.eval import EvalSpec, evaluate
from far2.train import Stage, Trainer
from far2.utils import pick_device, set_seed


def parser(desc: str, steps: int, eval_samples: int = 200) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--steps", type=int, default=steps)
    p.add_argument("--eval-samples", type=int, default=eval_samples)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--mini", action="store_true",
                   help="~10x smaller run for a quick sanity pass")
    return p


def apply_mini(args) -> None:
    if args.mini:
        args.steps = max(150, args.steps // 10)
        args.eval_samples = max(32, args.eval_samples // 5)
        args.batch = min(args.batch, 64)


def make_tc(args, name: str, **overrides) -> TrainConfig:
    # A 0.999 EMA retains 77.9% of random initialization after 250 updates and
    # makes mini checkpoints look untrained.  Keep the confirm-scale default,
    # but use a shorter-memory EMA for genuinely short diagnostic runs.
    ema = 0.99 if args.steps < 1000 else 0.999
    tc = TrainConfig(steps=args.steps, batch_size=args.batch, device=args.device,
                     seed=args.seed, out_dir=os.path.join(args.out, name),
                     warmup=max(50, args.steps // 15),
                     eval_samples=args.eval_samples, ema=ema)
    return replace(tc, **overrides)


def train(mc: ModelConfig, tc: TrainConfig, stages: list[Stage]) -> Trainer:
    from far2.models import build_model
    set_seed(tc.seed)              # BEFORE build_model: init is part of the seed
    model = build_model(mc)
    print(f"[{tc.out_dir}] {mc.kind} params={model.param_count():,} "
          f"device={pick_device(tc.device)} steps={tc.steps}")
    trainer = Trainer(model, mc, tc, stages)
    trainer.train()
    return trainer


def run_eval(trainer: Trainer, specs: list[EvalSpec], tag: str,
             seed: int = 1234) -> list[dict]:
    out_path = os.path.join(trainer.tc.out_dir, "eval", f"{tag}.jsonl")
    with trainer.eval_weights():
        return evaluate(trainer.model, specs, trainer.device,
                        seed=seed, out_path=out_path)


def finish(gate_ok: bool) -> None:
    sys.exit(0 if gate_ok else 1)
