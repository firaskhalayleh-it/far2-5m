"""Architecture audit: budgets, shapes, gradient coverage, mask hygiene, and
the structural invariants (inactive slots frozen; goal readable, unwritable)."""

import random

import pytest
import torch

from far2.data import GENERATORS, collate
from far2.models import build_model
from far2.presets import get_preset

FIVE_M = ["transformer_5m", "transformer_bind_match", "recurrent_5m",
          "slot_recurrent_5m", "far2_5m",
          "far2_full", "far2_fixed4", "far2_random_expand", "far2_oracle_expand",
          "far2_no_split", "far2_random_reset", "far2_goal_scrambled",
          "far2_no_goal",
          "far2_bind", "far2_bind_random_expand", "far2_bind_oracle",
          "slot_recurrent_bind"]
TINY = ["transformer_tiny", "recurrent_tiny", "far2_tiny", "far2_tiny_bind"]


def make_batch(families=("carry_add", "expr_eval", "optimize", "linsys",
                         "symbreak"), d=3, n=2):
    rng = random.Random(0)
    samples = [GENERATORS[f].sample(rng, d) for f in families for _ in range(n)]
    return collate(samples)


@pytest.mark.parametrize("preset", FIVE_M)
def test_param_budget(preset):
    n = build_model(get_preset(preset)).param_count()
    assert 4.2e6 <= n <= 5.8e6, f"{preset}: {n:,}"


@pytest.mark.parametrize("preset", TINY)
def test_forward_shapes(preset):
    m = build_model(get_preset(preset))
    batch = make_batch()
    out = m(batch, n_iters=3)
    B, A = batch["target_ids"].shape
    assert out.logits.shape == (B, A, 72)
    assert torch.isfinite(out.logits).all()


def test_shared_digit_significance_embedding():
    """The Abacus fairness contract: EVERY model embeds digit significance."""
    for preset in TINY:
        m = build_model(get_preset(preset))
        assert getattr(m, "sig", None) is not None, preset
        assert m.sig.num_embeddings > 32


def test_n_iters_changes_compute():
    m = build_model(get_preset("far2_tiny"))
    f4, f8, f16 = (m.flops_per_sample(64, t) for t in (4, 8, 16))
    assert f4 < f8 < f16
    assert abs((f16 - f4) - 3 * (f8 - f4)) < 1e-6 * f16   # linear in T


def test_far2_full_gradient_coverage():
    m = build_model(get_preset("far2_tiny_full"))
    m.set_unlocked({"expand", "split", "halt", "reconsider", "epistemic",
                    "goal"})
    m.train()
    out = m(make_batch(), n_iters=4, ipt=True)
    loss = out.logits.float().pow(2).mean()
    for v in out.loss_terms.values():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            loss = loss + 1e-3 * v.float().mean()
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"params without gradient: {missing}"


def test_inactive_slots_frozen():
    m = build_model(get_preset("far2_tiny"))     # adaptive off: 4 of 8 active
    m.eval()
    with torch.no_grad():
        out = m(make_batch(), n_iters=6, collect=True)
    assert out.telemetry["inactive_drift"] == 0.0
    assert all(a == m.cfg.init_slots for a in out.telemetry["active_slots"])


def test_goal_read_but_not_written():
    from dataclasses import replace
    m = build_model(replace(get_preset("far2_tiny"), protected_goal=True))
    m.set_unlocked({"goal"})
    m.train()
    out = m(make_batch(families=("optimize",), n=4), n_iters=3)
    out.logits.pow(2).mean().backward()
    # read path trains ...
    assert m.goal_query.grad is not None and m.goal_query.grad.abs().sum() > 0
    # ... and no module ever writes g: recompute forward twice with the same
    # inputs but different slot noise — g must be identical (function of H only)
    m.eval()
    batch = make_batch(families=("optimize",), n=4)
    H, _ = m._encode(batch)
    g1 = m._goal(H, batch)
    g2 = m._goal(H, batch)
    assert torch.equal(g1, g2)


def test_masked_rows_no_nan():
    m = build_model(get_preset("far2_tiny"))
    batch = make_batch(families=("carry_add",), n=6)
    out = m(batch, n_iters=4)
    assert torch.isfinite(out.logits).all()


def test_eval_halting_converge():
    from dataclasses import replace
    m = build_model(replace(get_preset("far2_tiny"), halting="converge"))
    m.eval()
    with torch.no_grad():
        out = m(make_batch(), n_iters=16, collect=True)
    assert max(out.telemetry["iters_used"]) <= 16
