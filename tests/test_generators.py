"""The ground-truth audit: every family's targets must match its independent
exact solver, samples must be deterministic in (seed, difficulty), and the
labels each phase relies on must be present — including the held-out transfer
families, the leak guard, and the D-vs-L decoupling v0.4 is built on."""

import random

import pytest

from far2.config import DERIVED, INCONSISTENT, OBSERVED, UNDERDET
from far2.data.base import collate, randomize_family_token
from far2.data.registry import (ALL_GENERATORS, GENERATORS,
                                TRANSFER_GENERATORS, Mix, MixSampler)
from far2.data.vocab import MAX_SIG, VOCAB, num_tokens, parse_num

FAMILIES = list(ALL_GENERATORS)


@pytest.mark.parametrize("family", FAMILIES)
def test_determinism(family):
    g = ALL_GENERATORS[family]
    d = (g.min_difficulty + g.max_difficulty) // 2
    a = g.sample(random.Random(7), d)
    b = g.sample(random.Random(7), d)
    assert a.tokens == b.tokens and a.targets == b.targets


@pytest.mark.parametrize("family", FAMILIES)
def test_solver_audit(family):
    """200 random samples per family across the difficulty range: the rendered
    context, replayed by an independent solver, must yield the stored targets."""
    g = ALL_GENERATORS[family]
    rng = random.Random(123)
    for i in range(200):
        d = rng.randint(g.min_difficulty, g.max_difficulty)
        s = g.sample(rng, d)
        assert g.solve(s) == s.targets, f"{family} d={d} sample {i}"
        for t in s.tokens + s.targets:
            assert t in VOCAB.ids, f"{family}: token {t} not in vocab"


@pytest.mark.parametrize("family", FAMILIES)
def test_sequences_fit(family):
    g = ALL_GENERATORS[family]
    rng = random.Random(5)
    for _ in range(50):
        s = g.sample(rng, g.max_difficulty)
        assert len(s.tokens) <= 512, f"{family}: {len(s.tokens)} tokens"


def test_depth_decoupled_from_length():
    """The load-bearing v0.4 property: carry DEPTH varies at FIXED digit
    count, so ∂Iter/∂D can be read with L controlled."""
    g = GENERATORS["carry_add"]
    rng = random.Random(2)
    lens = set()
    for d in (0, 8, 16):
        for _ in range(10):
            s = g.sample(rng, d, n_digits=24)
            lens.add(len(s.tokens))
            assert s.meta["D"] == d
    assert len(lens) == 1, f"token length varies with carry depth: {lens}"


def test_expr_depth_vs_tokens():
    """Expression-tree depth is the difficulty; token count must not be a
    monotone proxy for it (depth-2 wide trees vs depth-6 narrow chains)."""
    g = GENERATORS["expr_eval"]
    rng = random.Random(3)
    by_d = {d: [len(g.sample(rng, d).tokens) for _ in range(30)]
            for d in (2, 6)}
    assert max(by_d[2]) > min(by_d[6]), "token count separates depths perfectly"


def test_symbreak_pair_labels():
    g = GENERATORS["symbreak"]
    rng = random.Random(3)
    labels = set()
    for _ in range(100):
        s = g.sample(rng, 3)
        labels.add(s.requires_distinction)
        assert "pair" in s.meta and "vars" in s.meta
        ta, tb = s.meta["pair"]
        va, vb = s.meta["vars"][ta], s.meta["vars"][tb]
        if s.requires_distinction == 1:      # probe pair is a broken pair
            assert s.meta["R"] == 3 and va != vb
        else:                                # matched symmetric control
            assert s.meta["R"] == 0 and va == vb
    assert labels == {0, 1}


def test_linsys_statuses_and_epistemics():
    g = GENERATORS["linsys"]
    rng = random.Random(6)
    seen = set()
    for _ in range(200):
        s = g.sample(rng, 4)
        seen.add(s.meta["status"])
        if s.meta["status"] == "under":
            assert s.targets == ["UNDERDETERMINED"] and s.epistemic == UNDERDET
        elif s.meta["status"] == "inconsistent":
            assert s.targets == ["NO_SOLUTION"] and s.epistemic == INCONSISTENT
        else:
            assert s.epistemic in (DERIVED, OBSERVED)
    assert seen == {"unique", "under", "inconsistent"}


def test_extraneous_trap_and_control():
    g = GENERATORS["extraneous"]
    rng = random.Random(4)
    cases = [g.sample(rng, 5) for _ in range(100)]
    traps = [s for s in cases if s.reconsider_label == 1]
    controls = [s for s in cases if s.reconsider_label == 0]
    assert traps and controls
    for s in traps:                          # tempting root is extraneous
        assert s.targets[-1] != s.meta["attractor"]
    for s in controls:                       # tempting root is the answer
        assert s.targets[-1] == s.meta["attractor"]


def test_base_arith_counterfactual_bases():
    g = GENERATORS["base_arith"]
    rng = random.Random(9)
    for base in (5, 7, 10, 6, 9, 11):        # train AND held-out test bases
        s = g.sample(rng, base)
        assert "BASE" in s.tokens
        b_pos = s.tokens.index("BASE")
        assert parse_num([s.tokens[b_pos + 1]] if base <= 10
                         else s.tokens[b_pos + 1: b_pos + 2], base=11) is not None


def test_answer_width_never_leaks_magnitude():
    """Answer slot count may track VISIBLE surface size (operand length) but
    never the hidden value: at fixed surface it must be constant."""
    rng = random.Random(11)
    g = GENERATORS["carry_add"]
    widths = {len(g.sample(rng, 4, n_digits=8).targets) for _ in range(40)}
    assert len(widths) == 1, f"carry_add: answer widths {widths}"
    for family, d in (("expr_eval", 3), ("optimize", 4)):
        g = GENERATORS[family]
        widths = {len(g.sample(rng, d).targets) for _ in range(40)}
        assert len(widths) == 1, f"{family}: answer widths {widths}"


def test_residual_telemetry():
    g = GENERATORS["carry_add"]
    rng = random.Random(13)
    s = g.sample(rng, 3)
    assert g.residual(s, s.targets) == 0
    wrong = list(s.targets)
    wrong[-1] = "D9" if wrong[-1] != "D9" else "D8"
    assert g.residual(s, wrong) > 0


def test_family_token_leak_guard():
    """The leading token is shuffled per episode; telemetry keys off the name."""
    sampler = MixSampler([Mix("symbreak", 1.0, 2, 4)], seed=0)
    samples = sampler.batch(40)
    tokens = {s.tokens[0] for s in samples}
    assert len(tokens) > 3, "family token should vary across episodes"
    batch = collate(samples)
    from far2.data.base import FAMILY_ORDER
    assert batch["family_id"].unique().tolist() == [FAMILY_ORDER.index("symbreak")]
    s = MixSampler([Mix("symbreak", 1.0, 2, 4)], seed=0,
                   family_token_mode="true").one()
    assert s.tokens[0] == GENERATORS["symbreak"].family_token


def test_collate_variable_anchors():
    g = GENERATORS["symbreak"]
    rng = random.Random(17)
    samples = [g.sample(rng, 2) for _ in range(8)]
    batch = collate(samples)
    for b, s in enumerate(samples):
        for name, val in s.meta["vars"].items():
            vi = int(name[1:])
            pos = int(batch["var_pos"][b, vi])
            assert s.tokens[pos] == name
            if 0 <= val <= 9:
                assert int(batch["var_values"][b, vi]) == VOCAB.t(f"D{val}")
        pa, pb = s.meta["pair"]
        assert batch["pair_idx"][b].tolist() == [int(pa[1:]), int(pb[1:])]


def test_collate_sig_offset():
    g = GENERATORS["carry_add"]
    s = g.sample(random.Random(19), 2)
    plain = collate([s])["digit_sig"]
    off = collate([s], sig_offset=5)["digit_sig"]
    nz = plain > 0
    assert ((off[nz] - plain[nz]) == 5).all()
    assert (off[~nz] == 0).all()
    assert int(off.max()) <= MAX_SIG
    lo, hi = 10 ** 9, -1
    smp = MixSampler([Mix("carry_add", 1.0, 0, 2)], seed=0)
    for _ in range(50):
        o = smp.sig_offset()
        lo, hi = min(lo, o), max(hi, o)
    assert lo >= 0 and hi <= MAX_SIG - 34 and hi > 0


def test_transfer_families_never_trainable():
    with pytest.raises(AssertionError):
        MixSampler([Mix("sym_poly", 1.0, 2, 4)])


def test_transfer_conditions_present():
    rng = random.Random(11)
    sp = TRANSFER_GENERATORS["sym_poly"]
    labels = {sp.sample(rng, 3).requires_distinction for _ in range(60)}
    assert labels == {0, 1}
    rt = TRANSFER_GENERATORS["rational_trap"]
    cases = [rt.sample(rng, 4) for _ in range(60)]
    assert {s.reconsider_label for s in cases} == {0, 1}
    for s in cases:
        if s.reconsider_label == 1:          # excluded value: no solution
            assert s.targets == ["NO_SOLUTION"]
            assert s.meta["attractor"] is not None
    cg = TRANSFER_GENERATORS["congruence"]
    epis = [cg.sample(rng, 2).epistemic for _ in range(120)]
    assert epis.count(UNDERDET) > 5 and epis.count(DERIVED) > 5


def test_optimize_goal_span_and_violation():
    g = GENERATORS["optimize"]
    rng = random.Random(21)
    for _ in range(40):
        s = g.sample(rng, 4)
        assert "GOAL" in s.tokens and "Q" in s.tokens
        opt = parse_num(s.targets)
        if opt and opt > 0:
            below = num_tokens(opt - 1, width=len(s.targets))
            sc = g.score(s, below)
            assert sc["goal_violation"] == "infeasible_value"
            assert g.score(s, s.targets)["goal_violation"] is None
            break
    else:
        pytest.fail("no positive-optimum case found")
