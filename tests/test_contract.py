"""Scientific-contract tests — not "does it run" but "does it test what the
roadmap claims". Each test pins one review blocker closed:

  1-3  symbreak really demands R simultaneous bindings: length is
       DECORRELATED from R (live placebo filler axis, not constant L),
       every constraint row is necessary, the condition is invisible in
       the surface grammar, and the v0.4.1 streaming-scalar linear attack
       is fit directly and shown to fail
  4    autonomous training ends with ZERO oracle/binding supervision
  5    ablation arms train on IDENTICAL data (only unlocks differ)
  6    the held-out transfer families stay frozen during ablations
  7    initialization is a deterministic function of the declared seed
  8    evaluation is PAIRED across iteration counts T
  9    JSONL evidence files never mix reruns
"""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from far2.data.registry import GENERATORS, TRANSFER_GENERATORS
from far2.data.symbreak import N_PAIRS, solve_pairs
from far2.data.vocab import parse_num
from far2.models import build_model
from far2.presets import get_preset
from far2.utils import set_seed


def _rows(tokens):
    span = tokens[3: tokens.index("Q")]
    out, cur = [], []
    for t in span + ["SEP"]:
        if t == "SEP":
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    return out


def test_symbreak_length_decorrelated_from_R():
    """v0.4.2: L is a LIVE placebo axis (inert filler rows), decorrelated
    from R by construction — so the slots~(R,L) surface regression is
    testable and only R can carry the effect."""
    import statistics
    g = GENERATORS["symbreak"]
    rng = random.Random(0)
    Ls, Rs = [], []
    for _ in range(1500):
        s = g.sample(rng, rng.randint(1, 7))
        Ls.append(s.meta["L"])
        Rs.append(s.meta["R"])
    assert statistics.pstdev(Ls) > 0, "L constant: ols2 slots~(R,L) untestable"
    corr = statistics.correlation(Ls, Rs)
    assert abs(corr) < 0.12, f"L correlates with R: corr={corr:+.3f}"


def test_symbreak_every_pair_row_necessary():
    """No irrelevant distractors: dropping ANY pair's sum or product row
    leaves that pair's variables unresolved — all 8 pairs' constraints are
    simultaneously required (every pair is queried)."""
    g = GENERATORS["symbreak"]
    rng = random.Random(1)
    checked = 0
    for _ in range(25):
        s = g.sample(rng, 5)
        rows = _rows(s.tokens)
        for k, row in enumerate(rows):
            xs = {int(t[1:]) for t in row
                  if t.startswith("X") and t[1:].isdigit()}
            if not xs or "GEQ" in row:
                continue                    # filler / hint rows
            rebuilt = ["F5", "BOS", "CTX"]
            for j, r in enumerate(x for i, x in enumerate(rows) if i != k):
                if j:
                    rebuilt.append("SEP")
                rebuilt += r
            rebuilt += ["Q", "EOS"]
            got = solve_pairs(rebuilt)
            assert not (xs & set(got)), \
                f"dropping row {k} still resolves {xs & set(got)}"
            checked += 1
    assert checked >= 25 * 2 * N_PAIRS - 1


def test_symbreak_condition_invisible_in_surface():
    """Broken vs symmetric differs only in derived VALUES: the SET of
    reachable surface shapes (X* and digits collapsed) is identical across
    conditions — filler count varies per episode, never per condition."""
    g = GENERATORS["symbreak"]
    rng = random.Random(2)
    shapes = {0: set(), 1: set()}
    counts = {0: 0, 1: 0}
    shape = lambda s: tuple(sorted(
        ("X" if t.startswith("X") else
         "D" if (t.startswith("D") and len(t) == 2) else t)
        for t in s.tokens[1:]))
    while min(counts.values()) < 60:
        s = g.sample(rng, 4)
        shapes[s.requires_distinction].add(shape(s))
        counts[s.requires_distinction] += 1
    assert shapes[0] == shapes[1], \
        "surface grammar separates the conditions"


def test_symbreak_defeats_streaming_linear_attack():
    """Regression for the v0.4.1 collapse (d_q = Σ(3sᵢ−2tᵢ), one running
    scalar, 1400/1400 exact): fit the attack directly — least squares from
    every row constant + hint direction + query one-hots to the answer
    digits. On v0.4.2 (roots of x²−sx+p) it must fail; v0.4.1 scored 100%."""
    import numpy as np
    for name, dmax in (("symbreak", 7), ("sym_poly", 4)):
        g = (GENERATORS | TRANSFER_GENERATORS)[name]
        rng = random.Random(44)
        episodes = []
        for _ in range(700):
            s = g.sample(rng, rng.randint(1, dmax))
            f = {}
            for row in _rows(s.tokens):
                xs = tuple(sorted(t for t in row if t.startswith("X")
                                  and t[1:].isdigit()))
                if not xs:
                    continue
                if "GEQ" in row:
                    f[("geq",) + xs] = 1.0 if row[0] == max(xs) else -1.0
                else:
                    kind = ("pow" if "POW" in row
                            else "prod" if "TIMES" in row else "sum")
                    f[(kind,) + xs] = float(
                        parse_num(row[row.index("EQ") + 1:]))
            q0 = s.tokens.index("Q") + 1
            for slot, qv in enumerate(s.tokens[q0: s.tokens.index("ANS")]):
                f[("q", slot, qv)] = 1.0
            episodes.append((f, [int(t[1:]) for t in s.targets]))
        train, test = episodes[:550], episodes[550:]
        keys = sorted({k for f, _ in train for k in f},
                      key=lambda k: tuple(map(str, k)))
        kidx = {k: i for i, k in enumerate(keys)}
        def mat(rows):
            X = np.zeros((len(rows), len(keys) + 1))
            X[:, -1] = 1.0
            for r, (f, _) in enumerate(rows):
                for k, v in f.items():
                    if k in kidx:
                        X[r, kidx[k]] = v
            return X
        W, *_ = np.linalg.lstsq(mat(train),
                                np.array([y for _, y in train], float),
                                rcond=None)
        Yte = np.array([y for _, y in test], float)
        pred = np.clip(np.rint(mat(test) @ W), 0, 9)
        exact = float((pred == Yte).all(axis=1).mean())
        per_digit = float((pred == Yte).mean())
        assert exact < 0.40, f"{name}: linear attack episode-exact {exact:.1%}"
        assert per_digit < 0.70, f"{name}: linear attack per-digit {per_digit:.1%}"


def test_phase3b_final_stage_unsupervised():
    from run_phase3b import free_stages
    for substrate in ("standard", "bind"):
        last = free_stages(substrate)[-1]
        assert last.oracle_weight == 0.0, "oracle supervision must reach zero"
        assert last.bind_weight == 0.0, "binding curriculum must reach zero"


def test_ablation_curricula_data_matched():
    from run_ablations import capabilities, matched_stages
    arms = {n: get_preset(n) for n in
            ("far2_full", "slot_recurrent_5m", "transformer_5m", "far2_bind")}
    stages = {n: matched_stages(mc) for n, mc in arms.items()}
    ref = stages["far2_full"]
    for n, st in stages.items():
        assert len(st) == len(ref)
        for a, b in zip(st, ref):
            assert a.mixes == b.mixes, f"{n}: different training data"
            assert (a.frac, a.depth, a.oracle_weight, a.bind_weight) == \
                   (b.frac, b.depth, b.oracle_weight, b.bind_weight)
            assert a.unlocked <= b.unlocked  # only capability filtering
            assert a.unlocked == (b.unlocked & capabilities(arms[n]))
    assert not any(s.unlocked for s in stages["slot_recurrent_5m"])
    assert all(s.ipt_alpha == 0.0 for s in stages["transformer_5m"])


def test_ablation_holdout_frozen():
    from run_ablations import eval_grid
    default = {spec.family for spec in eval_grid(10)}
    assert default <= set(GENERATORS), "held-out families leaked into ablation"
    unfrozen = {spec.family for spec in eval_grid(10, include_transfer=True)}
    assert set(TRANSFER_GENERATORS) <= unfrozen


def test_init_is_seed_deterministic():
    set_seed(123)
    a = build_model(get_preset("far2_tiny")).state_dict()
    set_seed(123)
    b = build_model(get_preset("far2_tiny")).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a), \
        "same declared seed must produce identical initial parameters"
    set_seed(124)
    c = build_model(get_preset("far2_tiny")).state_dict()
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_harness_paired_across_T():
    from far2.eval import EvalSpec, evaluate
    m = build_model(get_preset("far2_tiny"))
    recs = evaluate(m, [EvalSpec("carry_add", [2], [2, 3], n_samples=6)],
                    torch.device("cpu"), seed=5)
    by_T = {}
    for r in recs:
        by_T.setdefault(r["n_iters"], []).append(r["target"])
    assert by_T[2] == by_T[3], \
        "T=2 and T=3 must see the IDENTICAL episode sequence (paired)"


def test_signal_window_excludes_post_action_state():
    """P4 integrity: the pre-action window must include every signal up to and
    including the iteration whose signal triggers the action (so an action at
    the LAST iteration changes nothing), and must exclude everything after an
    EARLIER action — post-action state cannot vote for its own trigger."""
    from far2.data import GENERATORS, collate
    m = build_model(get_preset("far2_tiny"))     # operatorless: never acts
    m.eval()
    rng = random.Random(9)
    batch = collate([GENERATORS["symbreak"].sample(rng, 3) for _ in range(6)])
    T = 5
    with torch.no_grad():
        nat = m(batch, n_iters=T, collect=True).telemetry
        late = m(batch, n_iters=T, collect=True,
                 intervene={"force": [("split", T - 1)]}).telemetry
        early = m(batch, n_iters=T, collect=True,
                  intervene={"force": [("split", 1)]}).telemetry
    for k in nat["signal_max_pre"]:
        n, l, e = (nat["signal_max_pre"][k], late["signal_max_pre"][k],
                   early["signal_max_pre"][k])
        assert all(abs(a - b) < 1e-5 for a, b in zip(n, l)), \
            f"{k}: acting at the last iteration must not change the window"
        assert all(x <= y + 1e-5 for x, y in zip(e, n)), \
            f"{k}: an earlier action must only shrink the window"


def test_random_control_arms_are_curriculum_matched():
    """The decisions-vs-noise comparisons are only valid when the ONLY
    difference between the arms is the decision source: identical stages,
    mixes, fractions, and loss weights."""
    from run_phase3b import free_stages, random_stages
    for substrate in ("standard", "bind"):
        assert random_stages(substrate) == free_stages(substrate)
    from run_phase5a import stages_for
    assert stages_for("random_reset") == stages_for("reconsider")


def test_binding_candidates_probe_vs_searched():
    """The declared-pair overlap needs ground truth at runtime, so it may
    only ever appear as a non-selectable _probe key; the runtime-legitimate
    binding candidate is the SEARCHED 'bind' signal carried in signals_pre."""
    from far2.eval.metrics import signal_aurocs
    recs = [{"requires_distinction": i % 2,
             "signals": {"sim": float(i), "bind": float(i % 7)},
             "signals_pre": {"sim": float(i), "bind": float(i % 7)},
             "binding": {"overlap": float(i % 5)},
             "expand_count": 0, "split_count": 0}
            for i in range(24)]
    out = signal_aurocs(recs)
    assert "bind" in out and "bind_overlap_probe" in out
    assert not any(k == "bind_overlap" or k.endswith("_post") for k in out)


def test_model_never_reads_declared_pair():
    """Review #3 P0-1 pinned closed: perturbing batch['pair_idx'] alone —
    tokens and targets identical — must not change logits or any collision
    telemetry. The declared pair is evaluation/probe metadata only."""
    m = build_model(get_preset("far2_tiny_bind"))
    m.set_unlocked({"expand", "split"})
    m.eval()
    from far2.data import GENERATORS, collate
    rng = random.Random(3)
    batch = collate([GENERATORS["symbreak"].sample(rng, 4) for _ in range(6)])
    pert = dict(batch)
    pert["pair_idx"] = torch.roll(batch["pair_idx"], 1, dims=0)
    with torch.no_grad():
        a = m(batch, n_iters=4, collect=True)
        b = m(pert, n_iters=4, collect=True)
    assert torch.equal(a.logits, b.logits)
    assert a.telemetry["C_max"] == b.telemetry["C_max"]
    assert a.telemetry["signal_max"]["bind"] == b.telemetry["signal_max"]["bind"]


def test_non_inferiority_boundary_safe():
    """Review #3 P0-2: NI must never pass from zero-variance boundaries."""
    from far2.eval.stats import non_inferiority
    ok, d = non_inferiority(0, 320, 0, 320, margin=0.01)
    assert not ok and "NOT-TESTABLE" in d
    ok, d = non_inferiority(320, 320, 320, 320, margin=0.01)
    assert not ok and "NOT-TESTABLE" in d
    ok, _ = non_inferiority(0, 320, 40, 320, margin=0.01)
    assert not ok, "clearly inferior arm must fail NI"
    ok, _ = non_inferiority(300, 320, 200, 320, margin=0.03)
    assert ok, "clearly superior arm must pass NI"
    ok, _ = non_inferiority(320, 320, 310, 320, margin=0.02)
    assert ok, "better arm at upper boundary must still be testable"


def test_campaign_fingerprint_tracks_package_and_upstream(tmp_path):
    """A completed step is only skippable if the whole imported far2 package
    AND its upstream checkpoints are unchanged — not just its own script."""
    from run_campaign import fingerprint
    up = tmp_path / "up"
    up.mkdir()
    st = {"script": "run_phase4.py", "argv": ["--run", str(up)]}
    fps = {fingerprint(st, "pkg_a"), fingerprint(st, "pkg_b")}
    (up / "ckpt.pt").write_bytes(b"weights v1")
    fps.add(fingerprint(st, "pkg_a"))
    (up / "ckpt.pt").write_bytes(b"weights v2")
    fps.add(fingerprint(st, "pkg_a"))
    assert len(fps) == 4, "fingerprint must react to package hash and ckpt bytes"


def test_jsonl_never_mixes_reruns(tmp_path):
    from far2.utils import JsonlLogger
    from far2.utils.logging import read_jsonl
    p = str(tmp_path / "log.jsonl")
    first = JsonlLogger(p)
    first.log({"run": 1})
    first.close()
    second = JsonlLogger(p)
    second.log({"run": 2})
    second.close()
    rows = read_jsonl(p)
    assert rows == [{"run": 2}], f"rerun appended instead of truncating: {rows}"
