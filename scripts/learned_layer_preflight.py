"""Learned v0.4.52 foundation preflight, run before aggregate smoke tests.

The task intentionally contains only two directly stated entities.  It does
not test FAR2's research benchmark or autonomous operators.  It asks whether
the learned encode -> entity -> warm workspace -> bind -> retain -> decode
path works when no mathematical difficulty can hide broken plumbing.
"""

import _common  # noqa: F401  (scripts path + repository import setup)
import argparse
import json
import os
import random
from dataclasses import replace

import torch
import torch.nn.functional as F

from far2.data import Sample, collate
from far2.data.vocab import VOCAB
from far2.models import build_model
from far2.presets import get_preset
from far2.train.loop import param_groups
from far2.utils import pick_device, set_seed


def _fact(rng: random.Random) -> Sample:
    a, b = rng.sample(range(10), 2)
    query = rng.randrange(2)
    tokens = ["F5", "BOS", "CTX", "X0", "EQ", f"D{a}", "SEP",
              "X1", "EQ", f"D{b}", "Q", f"X{query}", "ANS", "EOS"]
    values = {"X0": a, "X1": b}
    target = f"D{values[f'X{query}']}"
    return Sample("symbreak", 0, tokens, [target],
                  meta={"vars": values})


def _batch(rng, n, device):
    return collate([_fact(rng) for _ in range(n)], device)


def _answer_ce(out, batch, alpha=0.5):
    target = batch["target_ids"]
    vocab = out.logits.shape[-1]
    final = F.cross_entropy(out.logits.reshape(-1, vocab), target.reshape(-1),
                            ignore_index=-100)
    cycle = out.loss_terms["cycle_logits"]
    repeated = target[:, None, :].expand(-1, cycle.shape[1], -1)
    intermediate = F.cross_entropy(cycle.reshape(-1, vocab), repeated.reshape(-1),
                                   ignore_index=-100)
    loss = (1 - alpha) * final + alpha * intermediate
    for key in ("entity_nll", "bind_nll"):
        if key in out.loss_terms:
            loss = loss + 0.5 * out.loss_terms[key]
    return loss


def _gradient_groups(model, batch):
    model.zero_grad(set_to_none=True)
    out = model(batch, n_iters=4, all_cycle_logits=True)
    _answer_ce(out, batch).backward()
    names = {
        "encoder": ("encoder.", "enc_ln."),
        "initializer": ("cross_attn.",),
        "recurrent_core": ("self_attn.", "mlp.", "gate."),
        "binding": ("bind_W.", "entity_"),
        "decoder": ("dec_cross.", "dec_mlp.", "head."),
    }
    result = {}
    for group, prefixes in names.items():
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith(prefixes)]
        result[group] = bool(grads and all(g is not None for g in grads)
                             and sum(float(g.abs().sum()) for g in grads) > 0)
    model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def _evaluate(model, seed, n, depths=(1, 4, 8)):
    model.eval()
    batch = _batch(random.Random(seed), n, next(model.parameters()).device)
    target = batch["target_ids"]
    exact = {}
    final = None
    for depth in depths:
        out = model(batch, n_iters=depth, collect=(depth == 4))
        pred = out.logits.argmax(-1)
        exact[str(depth)] = float((pred == target).all(-1).float().mean())
        if depth == 4:
            final = out
    tel = final.telemetry
    present = tel["entity_present"]
    labels = batch["var_values"] - VOCAB.t("D0")
    entity_pred = tel["entity_value_logits"].argmax(-1)
    entity_acc = float((entity_pred[present] == labels[present]).float().mean())
    cycle_pred = tel["probe_preds"]
    cycle_acc = [float((cycle_pred[:, i] == target).all(-1).float().mean())
                 for i in range(cycle_pred.shape[1])]
    return {"exact": exact, "entity_accuracy": entity_acc,
            "cycle_exact": cycle_acc}


def run_learned_preflight(steps=300, batch_size=64, device="auto", seed=451,
                          out_dir="runs/learned_preflight"):
    dev = pick_device(device)
    cfg = replace(get_preset("far2_tiny_bind"), adaptive_slots=False,
                  halting="fixed", reconsider=False, epistemic=False,
                  protected_goal=False, init_slots=6, max_slots=8,
                  workspace_init="input")
    set_seed(seed)
    model = build_model(cfg).to(dev)
    rng = random.Random(seed)
    gradient = _gradient_groups(model, _batch(rng, batch_size, dev))
    opt = torch.optim.AdamW(param_groups(model, 0.01, 1e-3), lr=1e-3,
                            betas=(0.9, 0.95))
    losses = []
    model.train()
    for _ in range(steps):
        batch = _batch(rng, batch_size, dev)
        out = model(batch, n_iters=4, all_cycle_logits=True)
        loss = _answer_ce(out, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))

    warm = _evaluate(model, seed + 1000, 256)
    state = model.state_dict()
    shuffled = build_model(replace(cfg, workspace_init="input_shuffled")).to(dev)
    shuffled.load_state_dict(state)
    shuffled_metrics = _evaluate(shuffled, seed + 1000, 256)
    learned = build_model(replace(cfg, workspace_init="learned")).to(dev)
    learned.load_state_dict(state)
    learned_metrics = _evaluate(learned, seed + 1000, 256)

    first = sum(losses[:min(10, len(losses))]) / min(10, len(losses))
    last = sum(losses[-min(10, len(losses)):]) / min(10, len(losses))
    gates = {
        "L0_gradient_reachability": all(gradient.values()),
        "L1_optimization": last <= 0.35 * first,
        "L2_entity_extraction": warm["entity_accuracy"] >= 0.90,
        "L3_workspace_bind_decode": warm["exact"]["4"] >= 0.80,
        "L4_retention_T8": warm["exact"]["8"] >= warm["exact"]["4"] - 0.05,
        "L5_cycle_decodability": (len(warm["cycle_exact"]) == 4
                                   and warm["cycle_exact"][-1]
                                   >= warm["cycle_exact"][0] - 0.02),
        "L6_input_causality": (warm["exact"]["4"]
                                >= shuffled_metrics["exact"]["4"] + 0.20),
    }
    result = {
        "version": "0.4.52", "steps": steps, "batch_size": batch_size,
        "seed": seed, "device": str(dev), "params": model.param_count(),
        "loss_first": first, "loss_last": last, "gradient_groups": gradient,
        "warm": warm, "shuffled_init": shuffled_metrics,
        "learned_seed": learned_metrics, "gates": gates,
        "passed": all(gates.values()),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "learned_preflight.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("LEARNED-LAYER PREFLIGHT")
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"  loss {first:.4f} -> {last:.4f}")
    print(f"  warm exact T1/T4/T8: {warm['exact']}")
    print(f"  shuffled-init T4: {shuffled_metrics['exact']['4']:.1%}; "
          f"learned-seed T4: {learned_metrics['exact']['4']:.1%}")
    print(f"LEARNED-LAYER PREFLIGHT {'PASSED' if result['passed'] else 'FAILED'} "
          f"— {sum(gates.values())}/{len(gates)} gates")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=451)
    p.add_argument("--out", default="runs/learned_preflight")
    args = p.parse_args()
    result = run_learned_preflight(args.steps, args.batch, args.device,
                                   args.seed, args.out)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
