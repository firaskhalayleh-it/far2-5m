"""The 5M acceptance aggregator (docs/CRITERIA.md, final section) — the
mandatory-three-plus-two rule computed from CONFIRM-STAGE evidence across
seeds, never eyeballed:

  M1 computation adaptation    in-dist competence + depth grid + T16
                               stability + L-matched counterexample
  M2 representation adaptation phase3b quartet (vs fixed16 / random /
                               compute ratio rho / surface regression)
  M3 operator transfer         held-out sym_poly splits + rational_trap
plus >= 2 of: P5a reconsideration · P5b halting · P6 epistemics · P7 goal ·
interventions causal chain · R4 probes — every one judged from eval records
on disk with the same thresholds its phase gate uses; nothing hardcoded.

Discipline:
  · counts are POOLED across seeds for the tests (per-seed accuracy reported
    as mean ± sd alongside);
  · SEED COMPLETENESS is hard: evidence covering a subset of --seeds is
    NOT-TESTABLE — a lucky-seed subset must never masquerade as the claim;
  · anything missing or malformed on disk is NOT-TESTABLE — it can fail the
    rule but never pass it.

    python scripts/run_acceptance.py --out runs_confirm --seeds 0 1 2
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from far2.eval.metrics import assertion_counts, recovery_counts
from far2.eval.stats import (non_inferiority, ols2, rate_at_least,
                             rate_at_most, superiority, two_prop_test)
from far2.utils.logging import read_jsonl

HARD = {"carry_add": 8, "expr_eval": 8, "multiply": 4, "differentiate": 8}
TRAIN_T = 8                      # phase2 models train at T=8; 16 probes beyond


def load(out, pattern, seeds, tag):
    """[(seed, records)] for every runs dir that exists."""
    got = []
    for s in seeds:
        p = os.path.join(out, pattern.format(seed=s), "eval", f"{tag}.jsonl")
        if os.path.exists(p):
            got.append((s, read_jsonl(p)))
    return got


def complete(c, got, seeds, what) -> bool:
    """Hard seed-completeness enforcement."""
    if not got:
        c.not_testable(f"no {what} runs found")
        return False
    have = sorted(s for s, _ in got)
    missing = [s for s in seeds if s not in have]
    if missing:
        c.not_testable(f"{what}: seeds {missing} missing (have {have}) — "
                       "rerun them or drop them from --seeds")
        return False
    return True


def pooled(recs_by_seed, pred):
    k = n = 0
    per_seed = []
    for _, recs in recs_by_seed:
        sel = [r for r in recs if pred(r)]
        k += sum(r["correct"] for r in sel)
        n += len(sel)
        if sel:
            per_seed.append(sum(r["correct"] for r in sel) / len(sel))
    return k, n, per_seed


def pooled_counts(recs_by_seed, fn):
    """fn(records) -> (k, n); summed across seeds."""
    k = n = 0
    for _, recs in recs_by_seed:
        a, b = fn(recs)
        k += a
        n += b
    return k, n


def pooled_mean(recs_by_seed, key) -> float:
    vals = [r[key] for _, recs in recs_by_seed
            for r in recs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def seed_note(per_seed):
    if not per_seed:
        return "no seeds"
    m = statistics.mean(per_seed)
    sd = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
    return f"seeds n={len(per_seed)} mean={m:.1%} sd={sd:.1%}"


class Claim:
    def __init__(self, name):
        self.name, self.parts = name, []

    def check(self, label, ok, detail):
        self.parts.append((label, bool(ok), detail))
        return ok

    def not_testable(self, why):
        self.parts.append(("testable", False, f"NOT-TESTABLE: {why}"))

    @property
    def ok(self):
        return bool(self.parts) and all(ok for _, ok, _ in self.parts)

    def show(self):
        mark = "PASS" if self.ok else "FAIL"
        print(f"\n  [{mark}] {self.name}")
        for label, ok, detail in self.parts:
            print(f"     {'ok ' if ok else 'NO '} {label}: {detail}")


# ---- mandatory ------------------------------------------------------------
def m1_computation(out, seeds):
    c = Claim("M1 · computation adaptation (mandatory)")
    grid = load(out, "phase2_random_seed{seed}", seeds, "depth_grid")
    match = load(out, "phase2_random_seed{seed}", seeds, "length_match")
    if not complete(c, grid, seeds, "phase2 depth_grid"):
        return c
    hard = lambda r: r["difficulty"] >= HARD.get(r["family"], 99)
    # competence floor (P2 gate 1): OOD superiority on a model that never
    # learned the tasks (5% > 2%) is meaningless — in-distribution problems
    # are shallow, so they must clear 60% already at T=2
    ki, ni, pi = pooled(grid, lambda r: not hard(r) and r["n_iters"] == 2)
    ok, d = rate_at_least(ki, ni, point_min=0.60, ci_floor=0.55)
    c.check("in-dist competence at T=2 (>=60%)", ok, d + f" · {seed_note(pi)}")
    for lo, hi in ((2, 4), (4, 8)):
        k1, n1, p1 = pooled(grid, lambda r, hi=hi: hard(r) and r["n_iters"] == hi)
        k0, n0, _ = pooled(grid, lambda r, lo=lo: hard(r) and r["n_iters"] == lo)
        ok, d = superiority(k1, n1, k0, n0, margin=0.03)
        c.check(f"OOD acc T={hi} > T={lo} (+3pts)", ok, d + f" · {seed_note(p1)}")
    # stability past training depth (P2 gate 3, hard bucket): extra
    # iterations must not corrupt state
    k16, n16, _ = pooled(grid, lambda r: hard(r) and r["n_iters"] == 2 * TRAIN_T)
    k8, n8, _ = pooled(grid, lambda r: hard(r) and r["n_iters"] == TRAIN_T)
    ok, d = non_inferiority(k16, n16, k8, n8, margin=0.02)
    c.check(f"OOD T={2 * TRAIN_T} holds vs T={TRAIN_T} (NI 2pts)", ok, d)
    if complete(c, match, seeds, "phase2 length_match"):
        k0, n0, _ = pooled(match, lambda r: r["difficulty"] == 0)
        k24, n24, _ = pooled(match, lambda r: r["difficulty"] == 24)
        ok1, d1 = superiority(k0, n0, k24, n24, margin=0.05)
        ok2, d2 = rate_at_least(k0, n0, point_min=0.90, ci_floor=0.85)
        c.check("L-matched: length alone is not the cost", ok1 or ok2,
                f"sup: {d1} / high: {d2}")
    return c


def m2_representation(out, seeds, tagged):
    c = Claim("M2 · representation adaptation (mandatory)")
    adp = load(out, f"phase3b_adaptive{tagged}_seed{{seed}}", seeds, "phase3b")
    rnd = load(out, f"phase3b_random{tagged}_seed{{seed}}", seeds, "phase3b")
    f16 = load(out, f"phase3a_fixed16{tagged}_seed{{seed}}", seeds, "phase3a")
    if not complete(c, adp, seeds, "phase3b adaptive"):
        return c
    sym = lambda r: r["family"] == "symbreak"
    dq = lambda r: sym(r) and r.get("requires_distinction") == 1
    if complete(c, f16, seeds, "fixed16 control"):
        ka, na, pa = pooled(adp, sym)
        kf, nf, _ = pooled(f16, sym)
        ok, d = non_inferiority(ka, na, kf, nf, margin=0.01)
        c.check("adaptive NI fixed16 (1pt)", ok, d + f" · {seed_note(pa)}")
        # the allocation half of the claim: matching fixed16 is only
        # interesting at materially LOWER slot-compute
        sc_a = (pooled_mean(adp, "active_slots")
                * pooled_mean(adp, "iters_used"))
        sc_f = (pooled_mean(f16, "active_slots")
                * pooled_mean(f16, "iters_used"))
        rho = sc_a / sc_f if sc_f == sc_f and sc_f else float("nan")
        c.check("compute ratio rho <= 0.70", rho == rho and rho <= 0.70,
                f"rho={rho:.2f} ({sc_a:.1f} vs {sc_f:.1f} slot·iters)")
    if complete(c, rnd, seeds, "rate-matched random arm"):
        ka, na, _ = pooled(adp, dq)
        kr, nr, _ = pooled(rnd, dq)
        ok, d = superiority(ka, na, kr, nr, margin=0.03)
        c.check("adaptive > rate-matched random on distinction queries", ok, d)
    rows = [r for _, recs in adp for r in recs
            if sym(r) and r.get("active_slots") is not None
            and r.get("R") is not None and r.get("L") is not None]
    s = ols2([float(r["active_slots"]) for r in rows],
             [float(r["R"]) for r in rows], [float(r["L"]) for r in rows])
    slots_ok = (s.get("testable", False) and s["p1_pos"] < 0.05
                and (s["p2_pos"] >= 0.05 or s["beta2"] <= 0.5 * s["beta1"]))
    c.check("active slots track R, not L", slots_ok,
            f"beta_R={s.get('beta1', float('nan')):+.2f} "
            f"(p={s.get('p1_pos', float('nan')):.4f}) "
            f"beta_L={s.get('beta2', float('nan')):+.2f} "
            f"(p={s.get('p2_pos', float('nan')):.4f}) n={s.get('n', 0)}")
    return c


def m3_transfer(out, seeds, tagged):
    c = Claim("M3 · cross-domain operator transfer (mandatory)")
    far2 = []
    for cand in (f"ablate_far2_bind_seed{{seed}}",
                 f"phase3b_adaptive{tagged}_seed{{seed}}"):
        far2 = load(out, cand, seeds, "transfer")
        if far2:
            break
    ctrl = []
    for cand in ("ablate_slot_recurrent_bind_seed{seed}",
                 "ablate_slot_recurrent_5m_seed{seed}",
                 "phase0_slot_recurrent_seed{seed}"):
        ctrl = load(out, cand, seeds, "transfer")
        if ctrl:
            break
    if not complete(c, far2, seeds, "FAR2 transfer"):
        return c
    sp = [r for _, recs in far2 for r in recs if r["family"] == "sym_poly"]
    tp = sum(1 for r in sp if r.get("requires_distinction") == 1
             and (r.get("split_count") or 0) > 0)
    n_need = sum(1 for r in sp if r.get("requires_distinction") == 1)
    fp = sum(1 for r in sp if r.get("requires_distinction") == 0
             and (r.get("split_count") or 0) > 0)
    n_ctrl = sum(1 for r in sp if r.get("requires_distinction") == 0)
    ok, d = superiority(tp, n_need, fp, n_ctrl, margin=0.10)
    c.check("spontaneous splits track the unseen condition (+10pts)", ok, d)
    if complete(c, ctrl, seeds, "slot-recurrent control transfer"):
        rt = lambda recs: [r for r in recs if r["family"] == "rational_trap"
                           and r.get("reconsider_label") == 1]
        kf = sum(r["correct"] for _, recs in far2 for r in rt(recs))
        nf = sum(1 for _, recs in far2 for _r in rt(recs))
        kc = sum(r["correct"] for _, recs in ctrl for r in rt(recs))
        nc = sum(1 for _, recs in ctrl for _r in rt(recs))
        ok, d = superiority(kf, nf, kc, nc, margin=0.05)
        c.check("trap recovery beats slot-recurrent C (+5pts)", ok, d)
    return c


# ---- optional: judged from eval records with the phase gates' thresholds --
def opt_p5a(out, seeds):
    c = Claim("opt · P5a reconsideration at matched compute")
    arms = {a: load(out, f"phase5a_{a}_seed{{seed}}", seeds, "phase5a")
            for a in ("reconsider", "control", "random_reset")}
    t12 = load(out, "phase5a_control_seed{seed}", seeds, "phase5a_T12")
    if not all(complete(c, arms[a], seeds, f"phase5a {a}") for a in arms):
        return c
    rk = {a: pooled_counts(v, recovery_counts) for a, v in arms.items()}
    ok, d = superiority(*rk["reconsider"], *rk["control"], margin=0.05)
    c.check("recovery > control at identical compute (m=5pts)", ok, d)
    ok, d = superiority(*rk["reconsider"], *rk["random_reset"], margin=0.03)
    c.check("recovery > rate-matched random reset (m=3pts)", ok, d)
    if complete(c, t12, seeds, "phase5a control@T12"):
        k12, n12 = pooled_counts(t12, recovery_counts)
        ok, d = non_inferiority(*rk["reconsider"], k12, n12, margin=0.01)
        c.check("not worse than thinking longer (control@T12, NI 1pt)", ok, d)
    n_rec = pooled_mean(arms["reconsider"], "reconsider_count")
    c.check("trigger fires without collapsing", 0.05 <= n_rec <= 6,
            f"{n_rec:.2f} events/episode")
    ka, na, pa = pooled(arms["reconsider"], lambda r: True)
    kc, nc, _ = pooled(arms["control"], lambda r: True)
    ok, d = non_inferiority(ka, na, kc, nc, margin=0.02)
    c.check("accuracy not sacrificed (NI 2pts)", ok, d + f" · {seed_note(pa)}")
    return c


def opt_p5b(out, seeds):
    c = Claim("opt · P5b learned halting")
    lrn = load(out, "phase5b_learned_seed{seed}", seeds, "phase5b")
    cnv = load(out, "phase5b_learned_seed{seed}", seeds, "converge")
    if not (complete(c, lrn, seeds, "phase5b learned")
            and complete(c, cnv, seeds, "phase5b converge-rule control")):
        return c
    iters = pooled_mean(lrn, "iters_used")
    c.check("halting saves compute (mean iters < 7.5 of budget 8)",
            iters == iters and iters <= 7.5, f"{iters:.2f} mean iters")
    kl, nl, pl = pooled(lrn, lambda r: True)
    kc, nc, _ = pooled(cnv, lambda r: True)
    ok, d = non_inferiority(kl, nl, kc, nc, margin=0.02)
    c.check("accuracy NI the free convergence rule (2pts)", ok,
            d + f" · {seed_note(pl)}")
    rl, rn = pooled_counts(lrn, recovery_counts)
    cl, cn = pooled_counts(cnv, recovery_counts)
    ok, d = non_inferiority(rl, rn, cl, cn, margin=0.02)
    c.check("recovery maintained under halting (NI 2pts)", ok, d)
    return c


def opt_p6(out, seeds):
    c = Claim("opt · P6 mathematical epistemic state")
    epi = load(out, "phase6_epistemic_seed{seed}", seeds, "phase6")
    ctl = load(out, "phase6_control_seed{seed}", seeds, "phase6")
    if not (complete(c, epi, seeds, "phase6 epistemic")
            and complete(c, ctl, seeds, "phase6 control")):
        return c
    ku = nu = ko = no = 0
    for _, recs in epi:
        (a, b), (x, y) = assertion_counts(recs)
        ku, nu, ko, no = ku + a, nu + b, ko + x, no + y
    ok, d = rate_at_most(ku, nu, point_max=0.10, ci_ceiling=0.20)
    c.check("unsupported assertion <= 10%", ok, d)
    ok, d = rate_at_most(ko, no, point_max=0.10, ci_ceiling=0.20)
    c.check("over-abstention <= 10%", ok, d)
    ans = lambda r: r.get("is_unknown") is False
    ka, n1, pe = pooled(epi, ans)
    kc, n2, _ = pooled(ctl, ans)
    ok, d = non_inferiority(ka, n1, kc, n2, margin=0.02)
    c.check("answerable accuracy NI control (2pts)", ok, d + f" · {seed_note(pe)}")
    return c


def opt_p7(out, seeds):
    c = Claim("opt · P7 protected goal")
    pro = load(out, "phase7_protected_seed{seed}", seeds, "phase7")
    ctl = load(out, "phase7_control_seed{seed}", seeds, "phase7")
    if not (complete(c, pro, seeds, "phase7 protected")
            and complete(c, ctl, seeds, "phase7 control")):
        return c
    def viol(recs):
        sel = [r for r in recs if "goal_violation" in r]
        return (sum(r["goal_violation"] is not None for r in sel), len(sel))
    kp, np_ = pooled_counts(pro, viol)
    kc, nc = pooled_counts(ctl, viol)
    vp = kp / np_ if np_ else float("nan")
    vc = kc / nc if nc else float("nan")
    t = two_prop_test(kc, nc, kp, np_)
    c.check("violation <= 0.5 x control AND contrast p < 0.05",
            vp == vp and vc == vc and vp <= 0.5 * vc + 1e-9
            and t["p"] == t["p"] and t["p"] < 0.05,
            f"{vp:.1%} vs control {vc:.1%}, p={t['p']:.4f}")
    ka, na, pa = pooled(pro, lambda r: True)
    kb, nb, _ = pooled(ctl, lambda r: True)
    ok, d = non_inferiority(ka, na, kb, nb, margin=0.02)
    c.check("accuracy not sacrificed (NI 2pts)", ok, d + f" · {seed_note(pa)}")
    return c


def optional_from_summary(out, fname, name, judge):
    c = Claim(name)
    p = os.path.join(out, fname)
    if not os.path.exists(p):
        c.not_testable(f"{fname} not found")
        return c
    try:
        judge(c, json.load(open(p)))
    except Exception as e:  # malformed evidence is not evidence
        c.not_testable(f"unreadable summary: {e}")
    return c


def main():
    ap = argparse.ArgumentParser(description="confirm-stage acceptance rule")
    ap.add_argument("--out", default="runs_confirm",
                    help="root of CONFIRM-stage runs (not the pilot)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--substrate", choices=["standard", "bind"], default="bind")
    args = ap.parse_args()
    tagged = "" if args.substrate == "standard" else "_bind"
    print(f"[acceptance] root={args.out} seeds={args.seeds} "
          f"substrate={args.substrate}")
    if len(args.seeds) < 2:
        print("[acceptance] WARNING: fewer than 2 seeds — error bars are "
              "undefined; the verdict below is provisional at best")

    mandatory = [m1_computation(args.out, args.seeds),
                 m2_representation(args.out, args.seeds, tagged),
                 m3_transfer(args.out, args.seeds, tagged)]

    def judge_interventions(c, js):
        results = js.get("results", {})
        if not results:
            c.not_testable("summary has no per-operator results")
            return
        for op, r in results.items():
            v = r.get("verdict", "")
            usable = v.startswith("usage OK")
            c.check(f"{op} R8 usage", usable, v)
            if not usable:
                continue
            fu, us = r.get("false_usage"), r.get("usage")
            if fu == fu and us == us and fu is not None and us is not None:
                c.check(f"{op} selective usage P(use|cond) > P(use|no-cond)",
                        us > fu, f"{us:.1%} vs {fu:.1%}")
            if "necessity_k" in r:
                ok, d = rate_at_least(r["necessity_k"], r["necessity_n"],
                                      point_min=0.10, ci_floor=0.05)
                c.check(f"{op} necessity (blocking hurts)", ok, d)
            if r.get("placebo_n"):
                ok, d = rate_at_most(r["placebo_k"], r["placebo_n"],
                                     point_max=0.15, ci_ceiling=0.30)
                c.check(f"{op} placebo damage <= 15% (specificity)", ok, d)

    def judge_probes(c, js):
        r = js.get("run", {})
        if "error" in r or "bound" not in r:
            c.not_testable(r.get("error", "no probe result"))
            return
        c.check("R4 bound >= chance+20pts", r["bound"] >= r["chance"] + 0.20,
                f"bound {r['bound']:.1%} chance {r['chance']:.1%}")
        c.check("R4 separation >= 15pts + McNemar",
                r["separation_index"] >= 0.15 and r["mcnemar_p"] < 0.05,
                f"sep {r['separation_index']:+.1%} p={r['mcnemar_p']:.2e}")

    optional = [
        optional_from_summary(args.out, "interventions_summary.json",
                              "opt · do() causal chain", judge_interventions),
        optional_from_summary(args.out, "probes_summary.json",
                              "opt · R4 binding probes", judge_probes),
        opt_p5a(args.out, args.seeds),
        opt_p5b(args.out, args.seeds),
        opt_p6(args.out, args.seeds),
        opt_p7(args.out, args.seeds),
    ]

    print("\n================ ACCEPTANCE (confirm stage) ================")
    for c in mandatory + optional:
        c.show()
    n_m = sum(c.ok for c in mandatory)
    n_o = sum(c.ok for c in optional)
    verdict = n_m == 3 and n_o >= 2
    print(f"\n  mandatory: {n_m}/3 · optional: {n_o}/{len(optional)} "
          f"(need 3 + >=2)")
    print(f"  VERDICT: {'ACCEPTED at 5M' if verdict else 'NOT ACCEPTED'}"
          " — transfer is reported with seed error bars either way")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
