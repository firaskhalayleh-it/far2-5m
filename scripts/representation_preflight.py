"""Fast, untrained laws for FAR2's representation substrate.

This runs before the ordinary smoke test.  It proves mechanical contracts in
isolation; it does not claim that the learned task works.  A failed law is an
architecture failure and must stop training immediately.
"""

import _common  # noqa: F401  (scripts path + repository import setup)
import random
from dataclasses import replace

import torch

from far2.data import GENERATORS, collate
from far2.data.vocab import N_VARS
from far2.models import build_model
from far2.presets import get_preset
from far2.utils import set_seed


def _matched_symbreak(difficulty: int):
    rng = random.Random(41 + difficulty)
    need = control = None
    while need is None or control is None:
        sample = GENERATORS["symbreak"].sample(rng, difficulty)
        if sample.requires_distinction:
            need = sample
        else:
            control = sample
    return need, control


def _query_contract():
    need, _ = _matched_symbreak(4)
    batch = collate([need])
    expected = [int(t[1:]) for t in
                need.tokens[need.tokens.index("Q") + 1:need.tokens.index("ANS")]]
    assert batch["answer_query_vars"][0, :len(expected)].tolist() == expected
    assert batch["oracle_split_count"].item() == 4
    pairs = batch["oracle_split_pairs"][0, :4].tolist()
    assert len({tuple(p) for p in pairs}) == 4


def _workspace_only_queries():
    need, control = _matched_symbreak(2)
    batch = collate([need, control])
    set_seed(0)
    model = build_model(get_preset("far2_tiny_bind"))
    model.eval()
    with torch.no_grad():
        q = model._answer_queries(batch)
        perturbed = dict(batch)
        perturbed["input_ids"] = torch.roll(batch["input_ids"], 1, dims=0)
        perturbed["digit_sig"] = torch.roll(batch["digit_sig"], 1, dims=0)
        # Output queries are fixed by their explicit syntax tensors, not H.
        assert torch.equal(q, model._answer_queries(perturbed))

        S, d = model.cfg.max_slots, model.cfg.d_model
        z = torch.randn(2, S, d)
        active = torch.ones(2, S)
        a = model._decode(q, z, active)
        b = model._decode(q, torch.roll(z, 1, dims=0), active)
        assert not torch.allclose(a, b), \
            "decoder must be causally sensitive to workspace contents"


def _binding_and_collision_law():
    set_seed(1)
    model = build_model(get_preset("far2_tiny_bind"))
    S = model.cfg.max_slots
    B_mat = torch.zeros(1, N_VARS, S)
    B_mat[0, 0, 2] = 1.0
    B_mat[0, 1, 2] = 1.0
    present = torch.zeros(1, N_VARS)
    present[0, :2] = 1.0
    relation = torch.zeros(1, N_VARS, N_VARS)
    relation[0, 0, 1] = relation[0, 1, 0] = 1.0
    allowed = torch.zeros(1, S)
    allowed[0, 2:] = 1.0
    probs = torch.full((1, N_VARS, 10), 1e-6)
    probs[0, 0, 2] = 1.0
    probs[0, 1, 7] = 1.0
    probs = probs / probs.sum(-1, keepdim=True)
    collision = model._bind_collision(
        B_mat, probs, present, relation, allowed)
    score, source, pair, valid = collision[:4]
    assert valid.item() and score.item() > 0.5
    assert source.item() == 2 and set(pair[0].tolist()) == {0, 1}

    compatible = probs.clone()
    compatible[0, 1] = compatible[0, 0]
    no_collision = model._bind_collision(
        B_mat, compatible, present, relation, allowed)[0]
    assert no_collision.item() < 1e-6, \
        "shared ownership is a collision only for incompatible entities"


def _oracle_services_full_demand():
    need, control = _matched_symbreak(4)
    batch = collate([need, control])
    cfg = replace(get_preset("far2_tiny_bind"), adaptive_slots=True,
                  repr_policy="oracle", oracle_action="split",
                  init_slots=8, max_slots=16, n_iters_train=8)
    set_seed(2)
    model = build_model(cfg)
    model.set_unlocked({"expand", "split"})
    model.eval()
    with torch.no_grad():
        tel = model(batch, n_iters=8, collect=True).telemetry
    assert tel["split_count"] == [4.0, 0.0]
    # Two registers + eight base concepts + four requested splits.
    assert tel["active_slots"] == [14.0, 10.0]
    assignments = [x for x in tel["semantic_split_assignments"]
                   if x["batch"] == 0]
    assert len(assignments) == 4
    binding = tel["bind_matrix"][0]
    for row in assignments:
        x, y = row["pair"]
        sx, sy = row["slots"]
        assert sx != sy
        assert int(binding[x].argmax()) == sx
        assert int(binding[y].argmax()) == sy


def _free_policy_cannot_read_oracle_metadata():
    need, control = _matched_symbreak(3)
    batch = collate([need, control])
    cfg = replace(get_preset("far2_tiny_bind"), adaptive_slots=True,
                  init_slots=4, max_slots=8, repr_policy="learned")
    set_seed(3)
    model = build_model(cfg)
    model.set_unlocked({"expand", "split"})
    model.eval()
    perturbed = dict(batch)
    perturbed["oracle_split_pairs"] = torch.roll(
        batch["oracle_split_pairs"], 1, dims=0)
    perturbed["oracle_split_count"] = torch.roll(
        batch["oracle_split_count"], 1, dims=0)
    with torch.no_grad():
        a = model(batch, n_iters=5, collect=True)
        b = model(perturbed, n_iters=5, collect=True)
    assert torch.equal(a.logits, b.logits)
    assert a.telemetry["split_count"] == b.telemetry["split_count"]
    assert a.telemetry["C_max"] == b.telemetry["C_max"]


def _semantic_split_is_exclusive_and_equivariant():
    set_seed(4)
    model = build_model(get_preset("far2_tiny_bind"))
    d = model.cfg.d_model
    parent = torch.randn(5, d)
    ex = torch.randn(5, d)
    ey = torch.randn(5, d)
    scale = torch.full((5,), 0.2)
    children, assignment = model._semantic_split_children(parent, ex, ey, scale)
    swapped, swapped_assignment = model._semantic_split_children(
        parent, ey, ex, scale)
    assert torch.allclose(children[:, 0], swapped[:, 1], atol=1e-5)
    assert torch.allclose(children[:, 1], swapped[:, 0], atol=1e-5)
    assert torch.allclose(assignment.sum(-1), torch.ones(5, 2), atol=1e-4)
    assert torch.allclose(assignment.sum(-2), torch.ones(5, 2), atol=1e-4)
    assert torch.allclose(assignment, swapped_assignment.flip((1, 2)), atol=1e-4)
    assert torch.all((children[:, 0] - children[:, 1]).norm(dim=-1) > 1e-4)


def run_preflight():
    checks = [
        ("query + full-demand tensor contract", _query_contract),
        ("workspace-only output query path", _workspace_only_queries),
        ("binding compatibility/collision law", _binding_and_collision_law),
        ("oracle services every broken pair", _oracle_services_full_demand),
        ("free policy ignores oracle metadata", _free_policy_cannot_read_oracle_metadata),
        ("semantic SPLIT is exclusive + swap-equivariant",
         _semantic_split_is_exclusive_and_equivariant),
    ]
    print("REPRESENTATION PREFLIGHT")
    for name, check in checks:
        check()
        print(f"  [PASS] {name}")
    print(f"REPRESENTATION PREFLIGHT PASSED — {len(checks)}/{len(checks)} laws")


if __name__ == "__main__":
    run_preflight()
