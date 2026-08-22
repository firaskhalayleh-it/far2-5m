"""Acceptance-aggregator enforcement tests (review #3): the rule must be
impossible to satisfy with missing seeds, smoke-scale budgets, single seeds,
or checkpoints that cannot express the claimed operator."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_acceptance.py")


def run_acc(out, *extra):
    r = subprocess.run([sys.executable, SCRIPT, "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=ROOT)
    return r.returncode, r.stdout


def mk_run(root, name, tag, records, steps=6000, model=None):
    d = root / name / "eval"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{tag}.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    with open(root / name / "config.json", "w") as fh:
        json.dump({"model": model or {}, "train": {"steps": steps}}, fh)


def grid_records(correct=1):
    out = []
    for fam, dmax in (("carry_add", 12),):
        for d in (2, dmax):
            for T in (2, 4, 8, 16):
                out += [{"family": fam, "difficulty": d, "n_iters": T,
                         "correct": correct} for _ in range(30)]
    return out


def test_empty_root_is_not_accepted(tmp_path):
    rc, out = run_acc(tmp_path, "--seeds", "0", "1")
    assert rc == 1 and "NOT ACCEPTED" in out
    assert "NOT-TESTABLE" in out


def test_seed_incompleteness_is_listed(tmp_path):
    mk_run(tmp_path, "phase2_random_seed0", "depth_grid", grid_records())
    rc, out = run_acc(tmp_path, "--seeds", "0", "1")
    assert rc == 1
    assert "seeds [1] missing" in out


def test_smoke_budget_cannot_feed_acceptance(tmp_path):
    for s in (0, 1):
        mk_run(tmp_path, f"phase2_random_seed{s}", "depth_grid",
               grid_records(), steps=250)
    rc, out = run_acc(tmp_path, "--seeds", "0", "1")
    assert rc == 1
    assert "below confirm budget" in out and "250" in out


def test_single_seed_never_accepts(tmp_path):
    rc, out = run_acc(tmp_path, "--seeds", "0")
    assert rc == 1
    assert "fewer than 2 seeds" in out and "acceptance is impossible" in out


def test_m3_capability_validated_against_checkpoint(tmp_path):
    sp = [{"family": "sym_poly", "requires_distinction": i % 2,
           "split_count": 0, "correct": 0} for i in range(60)]
    model = {"adaptive_slots": True, "binding": True, "reconsider": False}
    for s in (0, 1):
        mk_run(tmp_path, f"ablate_far2_bind_seed{s}", "transfer", sp,
               model=model)
    rc, out = run_acc(tmp_path, "--seeds", "0", "1")
    assert rc == 1
    assert "lacks RECONSIDER" in out


def test_per_seed_summaries_required(tmp_path):
    with open(tmp_path / "interventions_summary_seed0.json", "w") as fh:
        json.dump({"results": {"split": {"verdict": "usage OK"}}}, fh)
    rc, out = run_acc(tmp_path, "--seeds", "0", "1")
    assert rc == 1
    assert "interventions_summary_seed<N>.json missing for seeds [1]" in out
