"""Audit whether a FAR2 smoke answer exists before recurrent computation.

Runs the same paired held-out multiplication cells at Z0, T=1, and T=4 with
the trained input initializer, learned seeds, and a shuffled input initializer.
This is a shortcut audit, not a benchmark.
"""

import _common  # noqa: F401
import argparse
import json
import os
import random
from dataclasses import replace

import torch

from far2.data.base import collate, randomize_family_token
from far2.data.registry import ALL_GENERATORS
from far2.models import build_model
from far2.train import Trainer


def _batches(device, n_samples, seed=1234):
    rng = random.Random(f"{seed}:multiply:1")
    gen = ALL_GENERATORS["multiply"]
    remaining = n_samples
    while remaining:
        n = min(64, remaining)
        remaining -= n
        samples = [gen.sample(rng, 1) for _ in range(n)]
        samples = [randomize_family_token(s, rng) for s in samples]
        yield collate(samples, device)


def _exact(pred, target):
    valid = target != -100
    return ((pred == target) | ~valid).all(-1)


def _z0_logits(model, batch):
    H, pad = model._encode(batch)
    Q = model._answer_queries(batch)
    B, S = batch["input_ids"].shape[0], model.cfg.max_slots
    active = torch.zeros(B, S, device=H.device)
    active[:, :model.cfg.n_registers + model.cfg.init_slots] = 1
    Z = model._init_workspace(H, pad, active)
    g = model._goal(H, batch)
    goal_token = model.goal_tok(g) if g is not None else None
    answer_binding = None
    if model.cfg.binding:
        entities, _ = model._entity_states(H, pad, batch)
        concept = active.clone()
        concept[:, :model.cfg.n_registers] = 0
        binding = model._bind_matrix(entities, Z, concept)
        answer_binding = model._answer_binding(binding, batch)
    return model._decode(Q, Z, active, goal_token, answer_binding)


@torch.no_grad()
def audit_checkpoint(checkpoint, device, n_samples):
    trained, cfg = Trainer.load_model(checkpoint, device=device)
    state = trained.state_dict()
    result = {}
    for mode in ("input", "learned", "input_shuffled"):
        model = build_model(replace(cfg, workspace_init=mode)).to(device)
        model.load_state_dict(state)
        model.eval()
        counts = {"n": 0, "z0": 0, "t1": 0, "t4": 0,
                  "cycles": [0, 0, 0, 0]}
        for batch in _batches(device, n_samples):
            target = batch["target_ids"]
            counts["n"] += target.shape[0]
            counts["z0"] += int(_exact(
                _z0_logits(model, batch).argmax(-1), target).sum())
            one = model(batch, n_iters=1, collect=True)
            counts["t1"] += int(_exact(one.logits.argmax(-1), target).sum())
            four = model(batch, n_iters=4, collect=True)
            counts["t4"] += int(_exact(four.logits.argmax(-1), target).sum())
            for t in range(4):
                counts["cycles"][t] += int(_exact(
                    four.telemetry["probe_preds"][:, t], target).sum())
        n = counts.pop("n")
        result[mode] = {
            k: ([x / n for x in v] if isinstance(v, list) else v / n)
            for k, v in counts.items()
        }
    warm = result["input"]
    result["verdict"] = {
        "z0_answer_laden": warm["z0"] >= 0.50,
        "flat_after_cycle1": warm["t4"] - warm["t1"] <= 0.02,
        "recurrent_reasoning_demonstrated": warm["t4"] - warm["t1"] >= 0.10,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs_v0451/smoke_final")
    p.add_argument("--out", default="runs_v0451/smoke_shortcut_audit.json")
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    rows = {}
    for name in ("far2_tiny", "far2_tiny_full", "far2_tiny_bind"):
        rows[name] = audit_checkpoint(
            os.path.join(args.root, name, "ckpt.pt"), device, args.samples)
        print(name, rows[name])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"source": args.root, "n_samples": args.samples,
                   "results": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
