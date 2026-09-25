"""Causal-control machinery (v0.3) + binding substrate (v0.4): random/oracle
controller policies, do() interventions, reset-at-random, disabled actions,
goal scrambling, shuffled epistemic labels, parameter parity of the control
presets, explicit binding, registers, and the R0/R1 probes. These are the
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


def test_oracle_policy_splits_every_broken_pair():
    m = build_model(adaptive_tiny(repr_policy="oracle"))
    m.set_unlocked({"expand", "split"})
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=6, collect=True).telemetry
    demand = batch["oracle_split_count"].tolist()
    for i, r in enumerate(demand):
        assert tel["split_count"][i] == float(r)


def test_oracle_action_expand_variant():
    m = build_model(adaptive_tiny(repr_policy="oracle", oracle_action="expand"))
    m.set_unlocked({"expand", "split"})
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        tel = m(batch, n_iters=6, collect=True).telemetry
    assert sum(tel["split_count"]) == 0
    demand = batch["oracle_split_count"].tolist()
    for i, r in enumerate(demand):
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
    plain.set_unlocked({"goal"})
    scr.set_unlocked({"goal"})
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


def test_capped_ownership_is_flagged_parameter_free_and_limits_slot_mass():
    base = get_preset("far2_tiny_bind")
    assert base.bind_norm == "softmax", "the default ownership law must not change"
    torch.manual_seed(0)
    plain = build_model(base)
    capped = build_model(replace(base, bind_norm="capped"))
    assert capped.param_count() == plain.param_count()
    capped.load_state_dict(plain.state_dict())

    d, S = base.d_model, base.max_slots
    E, Z = torch.randn(3, 16, d) * 4, torch.randn(3, S, d) * 4
    active = torch.ones(3, S)
    active[:, -1] = 0                             # 7 slots x 2 = 14 units of room
    present = torch.ones(3, 16, dtype=torch.bool)
    present[:, 9:] = False                        # 9 variables: the real 24:16 slack
    with torch.no_grad():
        soft = plain._bind_matrix(E, Z, active, None, present)
        cap = capped._bind_matrix(E, Z, active, None, present)
    # the default path ignores `present` and is the historical softmax
    assert torch.equal(soft, plain._bind_matrix(E, Z, active))
    for Bm in (soft, cap):
        assert torch.allclose(Bm.sum(-1), torch.ones(3, 16), atol=1e-4)
        assert float((Bm * (1 - active[:, None, :])).sum()) < 1e-4
    held = lambda Bm: (Bm * present[:, :, None]).sum(1).max()      # noqa: E731
    assert held(soft) > 2.7, "fixture must actually pile variables into one slot"
    # the default sweep count must make the cap HOLD even on sharp scores; a
    # hasty cap sheds only part of the pile-up (it let training collapse again)
    assert held(cap) < base.bind_capacity + 0.15
    hasty = build_model(replace(base, bind_norm="capped", bind_sweeps=2))
    hasty.load_state_dict(plain.state_dict())
    with torch.no_grad():
        assert held(hasty._bind_matrix(E, Z, active, None, present)) > held(cap)
    # more variables than capacity: the cap cannot hold, so load is balanced
    # instead and every variable still owns exactly one unit
    few = torch.zeros(3, S)
    few[:, :4] = 1                                # 4 x 2 < 12
    with torch.no_grad():
        tight = capped._bind_matrix(E, Z, few, None, present)
    assert torch.allclose(tight.sum(-1), torch.ones(3, 16), atol=1e-4)
    assert held(tight) < held(plain._bind_matrix(E, Z, few))
    # the cap is differentiable through the same two paths as the softmax
    capped.train()
    Bm = capped._bind_matrix(E.requires_grad_(), Z.requires_grad_(), active, None, present)
    Bm[:, :, 0].sum().backward()
    assert E.grad.abs().sum() > 0 and Z.grad.abs().sum() > 0


def test_capped_ownership_runs_end_to_end():
    m = build_model(replace(get_preset("far2_tiny_bind"), bind_norm="capped"))
    m.set_unlocked(set())
    m.eval()
    with torch.no_grad():
        tel = m(symbreak_batch(), n_iters=4, collect=True).telemetry
    B_mat = tel["bind_matrix"]
    assert torch.allclose(B_mat.sum(-1), torch.ones(B_mat.shape[:2]), atol=1e-4)
    assert float((B_mat * (1 - tel["active_mask"][:, None, :])).sum()) < 1e-4


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


# ---- encoder variants (flags; defaults unchanged) ---------------------------
def test_encoder_variants_are_flagged_and_syntax_only():
    base = get_preset("far2_tiny_bind")
    assert base.enc_pos == "absolute" and base.enc_clause_roles is False
    plain = build_model(base)
    rope = build_model(replace(base, enc_pos="rope"))
    assert rope.param_count() == plain.param_count(), "rotary positions add no weights"
    roles = build_model(replace(base, enc_clause_roles=True))
    batch = symbreak_batch()
    for m in (rope, roles):
        m.set_unlocked(set())
        m.eval()
        with torch.no_grad():
            assert torch.isfinite(m(batch, n_iters=2).logits).all()
    # clause features depend on token ids only: identical rows -> identical features
    ids = batch["input_ids"]
    feats = roles._clause_roles(ids)
    assert torch.equal(feats[0], roles._clause_roles(ids[:1])[0])
    # rotary encoding is relative: shifting every position leaves q.k unchanged
    from far2.models.layers import apply_rope, rope_tables
    cos, sin = rope_tables(12, 16, "cpu")
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    def dot(i, j):
        qi = apply_rope(q.expand(1, 1, 12, 16), cos, sin)[..., i, :]
        kj = apply_rope(k.expand(1, 1, 12, 16), cos, sin)[..., j, :]
        return float((qi * kj).sum())
    assert abs(dot(3, 4) - dot(7, 8)) < 1e-4


def test_slot_query_entities_starts_inert_and_only_steers_attention():
    base = replace(get_preset("far2_tiny_bind"), bind_norm="capped")
    torch.manual_seed(0)
    plain = build_model(base)
    routed = build_model(replace(base, slot_query_entities=True))
    routed.load_state_dict(plain.state_dict(), strict=False)
    for m in (plain, routed):
        m.set_unlocked(set())
        m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        assert torch.allclose(plain(batch, n_iters=3).logits,
                              routed(batch, n_iters=3).logits, atol=1e-5)
    # the entity vector may only reach the query projection of the cross-attention
    assert routed.param_count() - plain.param_count() == base.d_model ** 2
    import inspect
    src = inspect.getsource(type(routed).forward)
    assert "ent_q = self.slot_query_entity(" in src and "q_in = q_in + ent_q" in src
    assert src.count("slot_query_entity(") == 1


def test_clause_read_pools_rows_and_runs():
    base = replace(get_preset("far2_tiny_bind"), bind_norm="capped",
                   slot_read_clauses=True, slot_query_entities=True)
    m = build_model(base)
    m.set_unlocked(set())
    m.eval()
    batch = symbreak_batch()
    with torch.no_grad():
        H, _ = m._encode(batch)
        mem, mask = m._clause_memory(H, batch)
        tel = m(batch, n_iters=3, collect=True).telemetry
    ids = batch["input_ids"][0]
    seg = m.clause_sep_ids[ids].long().cumsum(0)
    keep = batch["pad_mask"][0]
    c = int(seg[keep][5])
    assert torch.allclose(mem[0, c], H[0][(seg == c) & keep].mean(0), atol=1e-5)
    assert torch.allclose(tel["clause_attn"].sum(-1), torch.ones_like(tel["clause_attn"].sum(-1)), atol=1e-4)
