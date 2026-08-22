"""The trainer trains: loss falls on a tiny task, checkpoints round-trip, and
sampling is reproducible."""

import os

import torch

from far2.config import TrainConfig
from far2.data.registry import Mix, MixSampler
from far2.data.base import collate
from far2.models import build_model
from far2.presets import get_preset
from far2.train import Stage, Trainer
from far2.utils.logging import read_jsonl

MIX = [Mix("carry_add", 1.0, 0, 2)]


def make_trainer(tmp_path, steps=80):
    mc = get_preset("far2_tiny")
    tc = TrainConfig(steps=steps, batch_size=32, lr=1e-3, warmup=10,
                     device="cpu", out_dir=str(tmp_path / "run"),
                     log_every=10, ema=0.99, seed=0)
    return Trainer(build_model(mc), mc, tc, [Stage("t", 1.0, MIX)])


def test_loss_decreases(tmp_path):
    trainer = make_trainer(tmp_path).train()
    log = read_jsonl(os.path.join(trainer.tc.out_dir, "train_log.jsonl"))
    first, last = log[0]["loss"], log[-1]["loss"]
    assert last < first * 0.75, f"loss {first:.3f} -> {last:.3f}"


def test_checkpoint_roundtrip(tmp_path):
    trainer = make_trainer(tmp_path, steps=15).train()
    sampler = MixSampler(MIX, seed=42)
    batch = collate(sampler.batch(8))
    with trainer.eval_weights():
        trainer.model.eval()
        with torch.no_grad():
            ref = trainer.model(batch, n_iters=3).logits
    loaded, _ = Trainer.load_model(os.path.join(trainer.tc.out_dir, "ckpt.pt"))
    loaded.eval()
    with torch.no_grad():
        got = loaded(batch, n_iters=3).logits
    assert torch.allclose(ref, got, atol=2e-4), \
        f"max diff {(ref - got).abs().max():.2e}"


def test_sampler_reproducible():
    a = MixSampler(MIX, seed=5).batch(20)
    b = MixSampler(MIX, seed=5).batch(20)
    assert [s.tokens for s in a] == [s.tokens for s in b]
