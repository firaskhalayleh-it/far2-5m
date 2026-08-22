"""Aggregations over harness records, plus the PASS/FAIL gate printer every
phase script uses. Nothing here re-derives ground truth — it only reads the
episode schema."""

from collections import defaultdict


def accuracy(records, **filters) -> float:
    sel = _filter(records, **filters)
    return sum(r["correct"] for r in sel) / len(sel) if sel else float("nan")


def correct_counts(records, **filters) -> tuple[int, int]:
    """(k, n) for the statistical gates (far2.eval.stats) — same filters as
    accuracy(), returned as counts so tests can be exact."""
    sel = _filter(records, **filters)
    return sum(r["correct"] for r in sel), len(sel)


def _filter(records, **filters):
    out = []
    for r in records:
        ok = True
        for k, v in filters.items():
            if k.endswith("_min"):
                ok &= r.get(k[:-4], -1e18) >= v
            elif k.endswith("_max"):
                ok &= r.get(k[:-4], 1e18) <= v
            else:
                ok &= r.get(k) == v
        if ok:
            out.append(r)
    return out


def acc_by(records, key: str) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key)].append(r["correct"])
    return {k: sum(v) / len(v) for k, v in sorted(groups.items(),
                                                  key=lambda kv: str(kv[0]))}


def mean(records, key: str, **filters) -> float:
    sel = [r[key] for r in _filter(records, **filters) if r.get(key) is not None]
    return sum(sel) / len(sel) if sel else float("nan")


def acc_per_flop(records, **filters) -> float:
    sel = _filter(records, **filters)
    if not sel:
        return float("nan")
    acc = sum(r["correct"] for r in sel) / len(sel)
    fl = sum(r["flops"] for r in sel) / len(sel)
    return acc / fl if fl else float("nan")


def split_selectivity(records) -> dict:
    """BEHAVIORAL correlation between SPLIT events and the task condition
    requires_distinction — a claim about when the model splits, not proof it
    internally disentangled anything (that needs binding_separation).
    Precision/recall are against the task condition."""
    sel = [r for r in records if r.get("requires_distinction", -1) >= 0
           and "split_count" in r]
    did = [r["split_count"] > 0 for r in sel]
    need = [r["requires_distinction"] == 1 for r in sel]
    tp = sum(d and n for d, n in zip(did, need))
    n_did, n_need = sum(did), sum(need)
    rate_when_required = (tp / n_need) if n_need else float("nan")
    fp = sum(d and not n for d, n in zip(did, need))
    n_ctrl = len(sel) - n_need
    return {"precision": (tp / n_did) if n_did else float("nan"),
            "recall": rate_when_required,
            "rate_when_required": rate_when_required,
            "rate_on_control": (fp / n_ctrl) if n_ctrl else float("nan"),
            # raw counts so gates can run exact statistics (far2.eval.stats)
            "tp": tp, "n_did": n_did, "n_need": n_need,
            "fp": fp, "n_ctrl": n_ctrl}


def binding_separation(records) -> dict:
    """Slot<->entity alignment (harness `binding` field): among episodes whose
    task required distinguishing a pair, did distinct slots carry the two
    entities' evidence — and did that co-occur with SPLIT events?"""
    sel = [r for r in records if r.get("binding")]
    if not sel:
        return {"n": 0}
    req = [r for r in sel if r.get("requires_distinction") == 1]
    ctrl = [r for r in sel if r.get("requires_distinction") == 0]
    split = [r for r in req if r.get("split_count", 0) > 0]
    frac = lambda rs: (sum(r["binding"]["separated"] for r in rs) / len(rs)
                       if rs else float("nan"))
    k = lambda rs: sum(r["binding"]["separated"] for r in rs)
    return {"n": len(sel),
            "separated_when_required": frac(req),
            "separated_on_control": frac(ctrl),
            "separated_after_split": frac(split),
            # raw counts so gates can run exact statistics (far2.eval.stats)
            "k_req": k(req), "n_req": len(req),
            "k_ctrl": k(ctrl), "n_ctrl": len(ctrl),
            "k_split": k(split), "n_split": len(split),
            "mean_selectivity": (sum(r["binding"]["selectivity_a"]
                                     + r["binding"]["selectivity_b"]
                                     for r in sel) / (2 * len(sel)))}


def recovery_counts(records) -> tuple[int, int]:
    """(k, n) over hooked episodes: the per-iteration probe answered the
    planted attractor at least once in the first half of reasoning."""
    hooked = [r for r in records
              if r.get("attractor_hit_at") and r.get("reconsider_label") == 1
              and min(r["attractor_hit_at"]) <= max(1, (r.get("iters_used") or 4) // 2)]
    return sum(r["correct"] for r in hooked), len(hooked)


def recovery_rate(records) -> float:
    """P(final correct | hooked by the attractor early)."""
    k, n = recovery_counts(records)
    return k / n if n else float("nan")


def assertion_counts(records) -> tuple[tuple[int, int], tuple[int, int]]:
    """((k_unsupported, n_unknowns), (k_over_abstain, n_answerable))."""
    unknowns = [r for r in records if r.get("is_unknown") is True]
    answerable = [r for r in records if r.get("is_unknown") is False]
    return ((sum(r["unsupported_assertion"] for r in unknowns), len(unknowns)),
            (sum(r["over_abstention"] for r in answerable), len(answerable)))


def assertion_rates(records) -> tuple[float, float]:
    (ku, nu), (ko, na) = assertion_counts(records)
    return (ku / nu if nu else float("nan"),
            ko / na if na else float("nan"))


def goal_violation_rate(records) -> float:
    sel = [r for r in records if "goal_violation" in r]
    return (sum(r["goal_violation"] is not None for r in sel) / len(sel)
            if sel else float("nan"))


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (Mann-Whitney U); ties get average rank."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2                     # 1-based average rank
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def signal_aurocs(records) -> dict[str, float]:
    """AUROC of each collision-signal summary against the requires_distinction
    task condition. PRE-ACTION windows are preferred when the records carry
    them: on an oracle model the post-action state trivially separates the
    conditions (the forced split IS the difference), so a detector tournament
    scored post-action would reward reading the effect, not the condition."""
    sel = [r for r in records
           if r.get("requires_distinction", -1) >= 0 and "signals" in r]
    if not sel:
        return {}
    out = {}
    use_pre = all("signals_pre" in r for r in sel)
    src = "signals_pre" if use_pre else "signals"
    labels = [r["requires_distinction"] for r in sel]
    for k in sel[0][src].keys():
        out[k] = auroc([r[src][k] for r in sel], labels)
    ckey = "C_max_pre" if use_pre and all("C_max_pre" in r for r in sel) \
        else "C_max"
    if all(ckey in r for r in sel):
        out["C_max_over_iters"] = auroc([r[ckey] for r in sel], labels)
    # binding-substrate candidate: ownership overlap of the declared pair —
    # measured on the FINAL state. Under a BLOCKED evaluation (no expand or
    # split ever fired: the P4 protocol) the final state IS pre-action and
    # the candidate is legitimately eligible; if any action fired, the key
    # says _post whenever the rest of the tournament is pre-action
    withb = [r for r in sel if r.get("binding", {}).get("overlap") is not None]
    if len(withb) >= 20:
        never_acted = all(not r.get("expand_count") and not r.get("split_count")
                          for r in sel)
        bk = ("bind_overlap" if (never_acted or not use_pre)
              else "bind_overlap_post")
        out[bk] = auroc(
            [r["binding"]["overlap"] for r in withb],
            [r["requires_distinction"] for r in withb])
    return out


def surface_slopes(records, ykey: str, xa: str = "D", xb: str = "L") -> dict:
    """Mathematical Generalization Surface regression: ykey (e.g. iters_used,
    active_slots) on (xa, xb) — the desired pattern is beta_xa > 0 significant
    with beta_xb ~ 0. Wraps stats.ols2 over records carrying all three keys."""
    from far2.eval.stats import ols2
    sel = [r for r in records
           if r.get(ykey) is not None and r.get(xa) is not None
           and r.get(xb) is not None]
    return ols2([float(r[ykey]) for r in sel],
                [float(r[xa]) for r in sel],
                [float(r[xb]) for r in sel])


def residual_monotonicity(records) -> dict:
    """Exact-progress telemetry: does the decoded candidate get mathematically
    closer to correct as iterations increase? Fraction of episodes with a
    non-increasing residual trajectory + the mean first-to-last drop."""
    trajs = [r["residual_traj"] for r in records
             if isinstance(r.get("residual_traj"), list)
             and len(r["residual_traj"]) >= 2]
    if not trajs:
        return {"n": 0}
    mono = [all(b <= a + 1e-9 for a, b in zip(t, t[1:])) for t in trajs]
    return {"n": len(trajs),
            "frac_monotone": sum(mono) / len(trajs),
            "mean_drop": sum(t[0] - t[-1] for t in trajs) / len(trajs)}


def state_norm_stats(records) -> dict:
    """Bounded-norm gate inputs: final-iteration RMS band and slope."""
    profiles = [r["state_norm"] for r in records if r.get("state_norm")]
    if not profiles:
        return {"min": float("nan"), "max": float("nan"), "slope": float("nan")}
    finals = [p[-1] for p in profiles]
    slopes = [(p[-1] - p[0]) / max(len(p) - 1, 1) for p in profiles if len(p) > 1]
    return {"min": min(finals), "max": max(finals),
            "slope": sum(slopes) / len(slopes) if slopes else 0.0}


# ---- gate printer ------------------------------------------------------
class Gate:
    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str) -> bool:
        self.rows.append((name, bool(passed), detail))
        return bool(passed)

    def report(self) -> bool:
        width = max(len(r[0]) for r in self.rows) if self.rows else 10
        print(f"\n=== GATE: {self.title} ===")
        for name, passed, detail in self.rows:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name:<{width}}  {detail}")
        ok = all(p for _, p, _ in self.rows)
        print(f"  => {'ALL PASS' if ok else 'GATE FAILED — stop here (see docs/ROADMAP.md)'}")
        return ok
