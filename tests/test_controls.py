"""Causal-control machinery (v0.3) + binding substrate (v0.4): random/oracle
controller policies, do() interventions, reset-at-random, disabled actions,
goal scrambling, shuffled epistemic labels, parameter parity of the control
presets, competitive binding, registers, and the R0/R1 probes. These are the
mechanisms that make a FAR² win hard to explain away."""

import random
from dataclasses import replace

import torch

from far2.data import GENERATORS, collate
from far2.eval.probes import (fit_linear_probe, pair_value_probe,
                              probe_accuracy, probe_selfcheck)
from far2.models import build_model
from far2.presets import (PARAM_IDENTICAL_TO_BIND, PARAM_IDENTICAL_TO_FULL,
                          get_preset)


def symbreak_batch(n=12, d=3, seed=3):
    rng = random.Random(seed)
    samples = [GENERATORS["symbreak"].sample(rng, d) for _ in range(n)]
    assert {s.requires_distinction for s in samples} == {0, 1}, "need both"
    return collate(samples)


def adaptive_tiny(**over):
    return replace(get_preset("far2_tiny"), adaptive_slots=True,
                   init_slots=2, max_slots=8, **over)


# ---- policies ----------------------------------------------------------
def test_random_policy_fires_and_replays_deterministically():
    m = build_model(adaptive_tiny(repr_policy="random", random_expand_p=0.5,
                                  random_split_p=0.3))
    m.set_unlocked({"expand", "split"})
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        a = m(batch, n_iters=6, collect=True).telemetry
        b = m(batch, n_iters=6, collect=True).telemetry
    total = sum(a["expand_count"]) + sum(a["split_count"])
    assert total > 0
    assert a["expand_count"] == b["expand_count"]      # eval draws are seeded
    assert a["split_count"] == b["split_count"]


def test_random_policy_zero_rate_is_inert():
    m = build_model(adaptive_tiny(repr_policy="random", random_expand_p=0.0,
                                  random_split_p=0.0))
    m.set_unlocked({"expand", "split"})
    m.eval()
    with torch.no_grad():
        tel = m(symbreak_batch(), n_iters=6, collect=True).telemetry
    assert sum(tel["expand_count"]) + sum(tel["split_count"]) == 0
    assert all(a == 2 for a in tel["active_slots"])


def test_oracle_policy_splits_exactly_on_the_condition():
    m = build_model(adaptive_tiny(repr_policy="oracle"))
    m.set_unlocked({"expand", "split"})
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=6, collect=True).telemetry
    req = batch["requires_distinction"].tolist()
    for i, r in enumerate(req):
        if r == 1:
            assert tel["split_count"][i] == 1.0
        else:
            assert tel["split_count"][i] == 0.0


def test_oracle_action_expand_variant():
    m = build_model(adaptive_tiny(repr_policy="oracle", oracle_action="expand"))
    m.set_unlocked({"expand", "split"})
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=6, collect=True).telemetry
    assert sum(tel["split_count"]) == 0
    req = batch["requires_distinction"].tolist()
    for i, r in enumerate(req):
        assert tel["expand_count"][i] == float(r)


def test_disable_actions_beats_unlock():
    m = build_model(adaptive_tiny(repr_policy="oracle",
                                  disable_actions=("split",)))
    m.set_unlocked({"expand", "split"})
    assert "split" not in m.unlocked
    m.eval()
    with torch.no_grad():
        tel = m(symbreak_batch(), n_iters=6, collect=True).telemetry
    assert sum(tel["split_count"]) == 0


# ---- do() interventions ------------------------------------------------
def test_intervene_block_split():
    m = build_model(adaptive_tiny(repr_policy="oracle"))
    m.set_unlocked({"expand", "split"})
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=6, collect=True,
                intervene={"block": {"split"}}).telemetry
    assert sum(tel["split_count"]) == 0


def test_intervene_force_split_on_operatorless_model():
    m = build_model(get_preset("far2_tiny"))       # adaptive off entirely
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        nat = m(batch, n_iters=5, collect=True).telemetry
        forced = m(batch, n_iters=5, collect=True,
                   intervene={"force": [("split", 2)]}).telemetry
    assert sum(nat["split_count"]) == 0
    assert all(c == 1.0 for c in forced["split_count"])
    assert all(a == m.cfg.init_slots + 1 for a in forced["active_slots"])


def test_intervene_force_reconsider_on_any_model():
    m = build_model(get_preset("far2_tiny"))       # no halt ctrl module
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=5, collect=True,
                intervene={"force": [("reconsider", 3)]}).telemetry
    assert all(c == 1.0 for c in tel["reconsider_count"])


# ---- reset-at-random ---------------------------------------------------
def test_random_reset_policy_rate():
    cfg = replace(get_preset("far2_tiny"), reconsider=True, halting="fixed",
                  reconsider_policy="random", random_reconsider_p=1.0)
    m = build_model(cfg)
    m.set_unlocked({"reconsider"})
    m.eval()
    with torch.no_grad():
        tel = m(symbreak_batch(), n_iters=5, collect=True).telemetry
    assert all(c == 5.0 for c in tel["reconsider_count"])
    m2 = build_model(replace(cfg, random_reconsider_p=0.0))
    m2.set_unlocked({"reconsider"})
    m2.eval()
    with torch.no_grad():
        tel2 = m2(symbreak_batch(), n_iters=5, collect=True).telemetry
    assert sum(tel2["reconsider_count"]) == 0


# ---- goal scramble -----------------------------------------------------
def test_goal_scrambled_is_a_batch_roll_with_identical_params():
    base = replace(get_preset("far2_tiny"), protected_goal=True)
    plain, scr = build_model(base), build_model(replace(base, goal_scrambled=True))
    scr.load_state_dict(plain.state_dict())
    assert plain.param_count() == scr.param_count()
    rng = random.Random(0)
    batch = collate([GENERATORS["optimize"].sample(rng, 3) for _ in range(4)])
    plain.eval(), scr.eval()
    with torch.no_grad():
        H, _ = plain._encode(batch)
        g_plain = plain._goal(H, batch)
        g_scr = scr._goal(H, batch)
    assert torch.allclose(g_scr, torch.roll(g_plain, 1, dims=0))


# ---- preset parity -----------------------------------------------------
def test_control_presets_param_identical_to_full():
    full = build_model(get_preset("far2_full")).param_count()
    for name in PARAM_IDENTICAL_TO_FULL:
        n = build_model(get_preset(name)).param_count()
        assert n == full, f"{name}: {n:,} != far2_full {full:,}"


def test_control_presets_param_identical_to_bind():
    bind = build_model(get_preset("far2_bind")).param_count()
    for name in PARAM_IDENTICAL_TO_BIND:
        n = build_model(get_preset(name)).param_count()
        assert n == bind, f"{name}: {n:,} != far2_bind {bind:,}"
    # slot_recurrent_bind is in that tuple: the C-vs-D comparison on the
    # binding substrate is now param-IDENTICAL (modules exist, operators
    # disabled), and it must be behaviorally inert as well
    sr = get_preset("slot_recurrent_bind")
    m = build_model(sr)
    m.set_unlocked({"expand", "split", "halt", "reconsider", "epistemic"})
    assert m.unlocked == set(), \
        "slot_recurrent_bind must never express a FAR2 operator"


# ---- binding substrate -------------------------------------------------
def test_bind_matrix_is_ownership_distribution():
    m = build_model(get_preset("far2_tiny_bind"))
    m.set_unlocked(set())
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=4, collect=True).telemetry
    B_mat = tel["bind_matrix"]
    Bsz, V, S = B_mat.shape
    assert V == 16 and S == m.cfg.max_slots
    assert torch.allclose(B_mat.sum(-1), torch.ones(Bsz, V), atol=1e-4)
    # mass only on active slots
    act = tel["active_mask"]
    assert float((B_mat * (1 - act[:, None, :])).sum()) < 1e-4
    assert tel["n_registers"] == m.cfg.n_registers == 2


def test_registers_always_active_no_actions():
    m = build_model(get_preset("far2_tiny_bind"))
    m.set_unlocked(set())
    m.eval()
    with torch.no_grad():
        tel = m(symbreak_batch(), n_iters=4, collect=True).telemetry
    want = m.cfg.init_slots + m.cfg.n_registers
    assert all(a == want for a in tel["active_slots"])


def test_bind_modules_receive_gradient():
    torch.manual_seed(0)
    m = build_model(replace(get_preset("far2_tiny_bind"), repr_policy="oracle"))
    m.set_unlocked({"expand", "split"})
    m.train()
    out = m(symbreak_batch(), n_iters=4, ipt=True)
    loss = out.logits.float().pow(2).mean()
    for v in out.loss_terms.values():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            loss = loss + 1e-3 * v.float().mean()
    assert "bind_nll" in out.loss_terms
    loss.backward()
    assert m.bind_W.weight.grad is not None and \
        m.bind_W.weight.grad.abs().sum() > 0
    assert m.split_dir_proj.weight.grad is not None, \
        "directed SPLIT projection must train through executed splits"


# ---- probes ------------------------------------------------------------
def test_linear_probe_recovers_separable_classes():
    g = torch.Generator().manual_seed(0)
    means = torch.randn(4, 16, generator=g) * 3
    y = torch.arange(4).repeat(50)
    X = means[y] + 0.3 * torch.randn(len(y), 16, generator=g)
    W = fit_linear_probe(X[:120], y[:120], 4)
    assert probe_accuracy(W, X[120:], y[120:]) > 0.9


def test_probe_selfcheck_passes_on_planted_encodings():
    sc = probe_selfcheck()
    assert sc["ok"], sc
    assert sc["planted"] > 0.95 and sc["shuffled"] < 0.25


def test_pair_value_probe_runs_on_tiny_models():
    for preset in ("far2_tiny", "far2_tiny_bind"):
        m = build_model(get_preset(preset))
        m.eval()
        out = pair_value_probe(m, torch.device("cpu"), n=48, difficulty=2, T=3)
        assert out["n_episodes"] >= 20, preset
        for k in ("bound", "wrong_slot", "pooled", "H_anchor", "bound_mlp",
                  "separation_index", "mcnemar_p", "diagnosis"):
            assert k in out, f"{preset}: missing {k}"
        assert 0.0 <= out["bound"] <= 1.0


# ---- shuffled epistemic labels ----------------------------------------
def test_epistemic_shuffled_control_trains(tmp_path):
    from far2.config import TrainConfig
    from far2.data.registry import Mix
    from far2.train import Stage, Trainer
    mc = replace(get_preset("far2_tiny"), epistemic=True)
    tc = TrainConfig(steps=3, batch_size=8, device="cpu", ema=0.0,
                     out_dir=str(tmp_path / "epi"), warmup=1, log_every=1,
                     epistemic_shuffled=True, epistemic_weight=0.5)
    g = GENERATORS["linsys"]
    stages = [Stage("only", 1.0, [Mix("linsys", 1.0, g.min_difficulty,
                                      g.min_difficulty + 1)],
                    unlocked={"epistemic"})]
    t = Trainer(build_model(mc), mc, tc, stages)
    t.train()
