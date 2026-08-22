"""The harness emits the telemetry schema (v0.4: D/R/L surface coordinates,
binding diagnostics, exact-residual trajectories) and the metric aggregations
match hand-computed values."""

import torch

from far2.eval import EvalSpec, evaluate
from far2.eval.metrics import (accuracy, assertion_rates, auroc, Gate,
                               goal_violation_rate, recovery_rate,
                               split_selectivity)
from far2.models import build_model
from far2.presets import get_preset

REQUIRED_KEYS = {"family", "difficulty", "correct", "pred", "target", "flops",
                 "D", "R", "L"}
FAR2_KEYS = {"iters_used", "active_slots", "expand_count", "split_count",
             "reconsider_count", "C_max", "signals"}


def test_harness_schema(tmp_path):
    m = build_model(get_preset("far2_tiny"))
    recs = evaluate(m, [EvalSpec("symbreak", [3], [3], n_samples=8),
                        EvalSpec("optimize", [3], [3], n_samples=8)],
                    torch.device("cpu"),
                    out_path=str(tmp_path / "eval.jsonl"))
    assert len(recs) == 16
    for r in recs:
        assert REQUIRED_KEYS <= set(r)
    sym = [r for r in recs if r["family"] == "symbreak"]
    assert all(FAR2_KEYS <= set(r) for r in sym)
    assert all("binding" in r for r in sym)      # slot<->variable diagnostics
    # surface coordinate recorded: R = 3 on condition episodes, 0 on controls
    assert all(r["R"] == (3 if r["requires_distinction"] == 1 else 0)
               for r in sym)
    assert all("goal_violation" in r for r in recs if r["family"] == "optimize")


def test_harness_residual_trajectory(tmp_path):
    """Exact per-iteration residuals from the probe decodes (model never sees
    them) must be present and end at 0 exactly when the episode is correct."""
    m = build_model(get_preset("far2_tiny"))
    recs = evaluate(m, [EvalSpec("carry_add", [2], [4], n_samples=8)],
                    torch.device("cpu"))
    for r in recs:
        assert "residual_traj" in r and len(r["residual_traj"]) == 4
        if r["correct"]:
            assert r["residual_traj"][-1] == 0


def test_harness_gen_kwargs(tmp_path):
    """EvalSpec.gen_kwargs reaches the generator (matched-counterexample
    machinery: 32-digit carry-free vs 24-chain)."""
    m = build_model(get_preset("far2_tiny"))
    recs = evaluate(m, [EvalSpec("carry_add", [0], [2], n_samples=4,
                                 gen_kwargs={"n_digits": 32})],
                    torch.device("cpu"))
    assert all(r["L"] > 60 for r in recs), "32-digit operands expected"
    assert all(r["D"] == 0 for r in recs)


def test_harness_reproducible():
    m = build_model(get_preset("far2_tiny"))
    m.eval()
    a = evaluate(m, [EvalSpec("expr_eval", [3], [2], n_samples=8)],
                 torch.device("cpu"), seed=9)
    b = evaluate(m, [EvalSpec("expr_eval", [3], [2], n_samples=8)],
                 torch.device("cpu"), seed=9)
    assert [r["target"] for r in a] == [r["target"] for r in b]


def test_auroc_hand_values():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-9


def test_metric_aggregations():
    recs = [
        {"family": "x", "difficulty": 2, "correct": True,
         "requires_distinction": 1, "split_count": 1.0},
        {"family": "x", "difficulty": 8, "correct": False,
         "requires_distinction": 0, "split_count": 1.0},
        {"family": "x", "difficulty": 8, "correct": True,
         "requires_distinction": 1, "split_count": 0.0},
        {"family": "u", "difficulty": 2, "correct": False, "is_unknown": True,
         "unsupported_assertion": True, "over_abstention": False},
        {"family": "u", "difficulty": 2, "correct": True, "is_unknown": False,
         "unsupported_assertion": False, "over_abstention": False},
        {"family": "g", "difficulty": 2, "correct": False,
         "goal_violation": "infeasible_value"},
        {"family": "g", "difficulty": 2, "correct": True, "goal_violation": None},
        {"family": "f", "difficulty": 2, "correct": True, "reconsider_label": 1,
         "attractor_hit_at": [0, 1], "iters_used": 6},
        {"family": "f", "difficulty": 2, "correct": False, "reconsider_label": 1,
         "attractor_hit_at": [1], "iters_used": 6},
    ]
    assert accuracy(recs, family="x") == 2 / 3
    assert accuracy(recs, difficulty_min=8) == 0.5
    sel = split_selectivity(recs)
    assert sel["precision"] == 0.5 and sel["recall"] == 0.5
    assert sel["rate_on_control"] == 1.0
    unsupported, over = assertion_rates(recs)
    assert unsupported == 1.0 and over == 0.0
    assert goal_violation_rate(recs) == 0.5
    assert recovery_rate(recs) == 0.5


def test_surface_slopes_recover_planted_effect():
    """ols2 wiring: iterations rise with D and are flat in L by construction."""
    from far2.eval.metrics import surface_slopes
    recs = []
    for D in range(6):
        for L in (40, 80, 120):
            for rep in range(8):
                recs.append({"family": "x", "difficulty": D, "correct": True,
                             "D": D, "R": 0, "L": L,
                             "iters_used": 2 + 2 * D + (rep % 2),
                             "active_slots": 4})
    s = surface_slopes(recs, "iters_used", "D", "L")
    assert s["testable"] and s["p1_pos"] < 0.05    # ∂Iter/∂D > 0
    assert abs(s["beta2"]) < 0.2                   # L contributes ~nothing


def test_gate_reports():
    g = Gate("t")
    g.check("a", True, "ok")
    g.check("b", False, "bad")
    assert g.report() is False
    g2 = Gate("t2")
    g2.check("a", True, "ok")
    assert g2.report() is True
