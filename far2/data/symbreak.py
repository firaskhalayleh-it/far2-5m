"""M5 symbreak — the flagship representation task, v0.4.2 (third redesign,
after the streaming-scalar collapse was proven on v0.4.1):

  · Every pair is constrained by SUM and PRODUCT — {X+Y=s, X·Y=p} — so a
    pair's values are the roots of x² − sx + p: recovering them is NONLINEAR
    in the given constants. No linear functional of the RHS stream computes
    any answer (v0.4.1 collapsed to d_q = Σ(3sᵢ−2tᵢ), one running scalar;
    a contract test now fits that linear attack directly and asserts it
    fails).
  · A late GEQ hint row per pair resolves WHICH variable is the larger root.
    For symmetric pairs (a=b) the hint is trivially true — the surface
    grammar is identical; broken-ness lives only in the derived values.
  · EVERY pair is queried (one side each, order announced only at the end):
    8 answers per episode, so no pair is a per-episode distractor and no
    aggregate statistic substitutes for the 8 individually-bound values —
    R of which additionally require a resolved root ORDER.
  · Inert constant-identity filler rows (no variables) vary sequence length
    INDEPENDENTLY of R: the slots~(R,L) surface regression gets a live,
    placebo L axis with corr(L,R) ≈ 0 by construction.

R = broken pairs (values differ) = simultaneously live root-order bindings.
requires_distinction = 1 iff R > 0; matched controls (same difficulty label,
same grammar, filler drawn identically) have all pairs symmetric. Honest
scope note: with d-dimensional slots no task can information-theoretically
FORCE slot growth — this task creates binding pressure; whether it
translates into capacity demand is exactly what the R2 quartet measures."""

import math
import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import num_tokens, parse_num

N_PAIRS = 8                              # 16 variables — always all of X0..X15
MAX_FILLER = 6


def _pair_rows(a: int, b: int, va: str, vb: str, rng: random.Random):
    sum_row = [va, "PLUS", vb, "EQ"] + num_tokens(a + b, width=2)
    prod_row = [va, "TIMES", vb, "EQ"] + num_tokens(a * b, width=2)
    if a == b:
        hi, lo = (va, vb) if rng.random() < 0.5 else (vb, va)
    else:
        hi, lo = (va, vb) if a > b else (vb, va)
    hint_row = [hi, "GEQ", lo]
    return sum_row, prod_row, hint_row


def _filler_row(rng: random.Random) -> list[str]:
    x, y = rng.randint(0, 9), rng.randint(0, 9)
    return [f"D{x}", "TIMES", f"D{y}", "EQ"] + num_tokens(x * y, width=2)


class SymBreakGenerator(TaskGenerator):
    name = "symbreak"
    family_token = "F5"
    min_difficulty = 1
    max_difficulty = 7                  # broken pairs among the 8

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        # matched control with p=0.5: identical surface, zero broken pairs
        condition = rng.random() < 0.5
        R = difficulty if condition else 0
        broken = set(rng.sample(range(N_PAIRS), R))
        vals: dict[int, tuple[int, int]] = {}
        for i in range(N_PAIRS):
            if i in broken:
                a = rng.randint(0, 9)
                b = rng.choice([v for v in range(10) if v != a])
            else:
                a = b = rng.randint(0, 9)
            vals[i] = (a, b)

        early, late = [], []
        for i in range(N_PAIRS):
            s_row, p_row, h_row = _pair_rows(*vals[i], f"X{2 * i}",
                                             f"X{2 * i + 1}", rng)
            early += [s_row, p_row]
            late.append(h_row)          # order information arrives LATE
        # inert filler: the placebo length axis, independent of R
        n_filler = rng.randint(0, MAX_FILLER)
        early += [_filler_row(rng) for _ in range(n_filler)]
        rng.shuffle(early)
        rng.shuffle(late)

        sides = [rng.randrange(2) for _ in range(N_PAIRS)]
        order = list(range(N_PAIRS))
        rng.shuffle(order)              # which pair answers where: revealed
        q_vars = [f"X{2 * i + sides[i]}" for i in order]      # only at the end

        tokens = [self.family_token, "BOS", "CTX"]
        for k, row in enumerate(early + late):
            if k:
                tokens.append("SEP")
            tokens += row
        tokens += ["Q"] + q_vars + ["ANS"] * N_PAIRS + ["EOS"]
        targets = [f"D{vals[i][sides[i]]}" for i in order]

        probe_pair = (min(broken) if broken
                      else rng.choice(range(N_PAIRS)))
        return Sample(
            self.name, difficulty, tokens, targets,
            epistemic=DERIVED,
            requires_distinction=int(condition),
            meta={"D": 2, "R": R, "L": len(tokens),
                  "pair": (f"X{2 * probe_pair}", f"X{2 * probe_pair + 1}"),
                  # Oracle-only upper-bound targets.  Free controllers never
                  # read this metadata; they must discover the same pairs from
                  # rendered relations and internal state.  Keeping every
                  # broken pair is essential: one episode can require up to
                  # six independent SPLIT actions, not merely one binary fire.
                  "split_pairs": [(f"X{2 * i}", f"X{2 * i + 1}")
                                  for i in sorted(broken)],
                  "vars": {f"X{2 * i + s}": vals[i][s]
                           for i in range(N_PAIRS) for s in (0, 1)}})

    def solve(self, sample: Sample) -> list[str]:
        vals = solve_pairs(sample.tokens)
        toks = sample.tokens
        q0 = toks.index("Q") + 1
        out = []
        for t in toks[q0: toks.index("ANS")]:
            vi = int(t[1:])
            v = vals.get(vi)
            assert v is not None, f"variable {t} unresolved"
            out.append(f"D{v}")
        return out

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        tgt = sample.targets
        p = (pred_tokens + ["PAD"] * len(tgt))[: len(tgt)]
        return float(sum(a != b for a, b in zip(p, tgt)))


def solve_pairs(tokens: list[str]) -> dict[int, int]:
    """Independent exact solver: per pair, roots of x² − sx + p ordered by
    the GEQ hint. Returns {var_index: value}; missing/inconsistent info for a
    pair leaves its variables unresolved."""
    span = tokens[3: tokens.index("Q")]
    rows, cur = [], []
    for t in span + ["SEP"]:
        if t == "SEP":
            if cur:
                rows.append(cur)
            cur = []
        else:
            cur.append(t)
    sums: dict[frozenset, int] = {}
    prods: dict[frozenset, int] = {}
    geq: dict[frozenset, tuple[int, int]] = {}
    for row in rows:
        xs = [int(t[1:]) for t in row if t.startswith("X") and t[1:].isdigit()]
        if not xs:
            continue                    # inert filler
        key = frozenset(xs)
        if "GEQ" in row:
            hi = int(row[0][1:])
            lo = int(row[2][1:])
            geq[key] = (hi, lo)
        else:
            rhs = parse_num(row[row.index("EQ") + 1:])
            (prods if "TIMES" in row else sums)[key] = rhs
    out: dict[int, int] = {}
    for key, s in sums.items():
        if key not in prods or key not in geq:
            continue
        p = prods[key]
        disc = s * s - 4 * p
        if disc < 0:
            continue
        r = math.isqrt(disc)
        if r * r != disc or (s + r) % 2:
            continue
        big, small = (s + r) // 2, (s - r) // 2
        hi, lo = geq[key]
        out[hi], out[lo] = big, small
    return out
