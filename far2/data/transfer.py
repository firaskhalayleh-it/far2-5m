"""Held-out transfer families (v0.4) — the hidden mathematical benchmark.
Each embeds a TRAINED cognitive condition inside an UNSEEN surface grammar;
the sampler structurally refuses to train on them (registry enforces it) and
the family token is shuffled everywhere, so nothing announces the condition.

  sym_poly       symmetry breaking via SYMMETRIC POLYNOMIAL constraints
                 (u+v, u*v — a grammar M5 never uses)          -> SPLIT
  rational_trap  excluded-value trap: (x^2 - a^2)/(x - a) = b simplifies to
                 x = b - a, INVALID when b = 2a (x = a is excluded)
                                                               -> RECONSIDER
  iter_apply     k-fold application of a declared affine map   -> recurrence
                 f(x) = c*x + e, unseen family, depth = k
  congruence     modular constraints pin X in 0..9 uniquely, ambiguously, or
                 not at all                                    -> epistemics
"""

import random

from far2.config import DERIVED, INCONSISTENT, UNDERDET
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import num_tokens, parse_num


class SymPolyGenerator(TaskGenerator):
    """v0.4.2 mirror of the redesigned symbreak in an UNSEEN grammar:
    POWER-SUM constraints {X²+Y² = s₂, X+Y = s₁} per pair (M5 uses
    sum+product; this uses square-sums — different tokens, same cognitive
    condition), a late GEQ hint that is load-bearing only when the pair is
    broken, ALL pairs queried (no per-episode distractors), and inert
    variable-free filler as the placebo length axis. The answer requires the
    roots of x² − s₁x + (s₁²−s₂)/2 — no linear functional of the constants
    computes it."""

    name = "sym_poly"
    family_token = "F0"                 # shuffled anyway; never trained on
    min_difficulty = 1
    max_difficulty = 4                  # broken pairs among the 5

    N_PAIRS = 5

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        n = self.N_PAIRS
        condition = rng.random() < 0.5
        R = difficulty if condition else 0
        broken = set(rng.sample(range(n), R))
        vals = []
        for i in range(n):
            if i in broken:
                u = rng.randint(0, 9)
                v = rng.choice([x for x in range(10) if x != u])
            else:
                u = v = rng.randint(0, 9)
            vals.append((u, v))

        early, late = [], []
        for i, (u, v) in enumerate(vals):
            a, b = f"X{2 * i}", f"X{2 * i + 1}"
            early.append([a, "POW", "D2", "PLUS", b, "POW", "D2", "EQ",
                          *num_tokens(u * u + v * v, width=3)])
            early.append([a, "PLUS", b, "EQ", *num_tokens(u + v, width=2)])
            if u == v:
                hi, lo = (a, b) if rng.random() < 0.5 else (b, a)
            else:
                hi, lo = (a, b) if u > v else (b, a)
            late.append([hi, "GEQ", lo])
        for _ in range(rng.randint(0, 4)):          # inert filler: L axis
            k = rng.randint(0, 9)
            early.append([f"D{k}", "POW", "D2", "EQ",
                          *num_tokens(k * k, width=2)])
        rng.shuffle(early)
        rng.shuffle(late)

        sides = [rng.randrange(2) for _ in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        q_vars = [f"X{2 * i + sides[i]}" for i in order]
        tokens = [self.family_token, "BOS", "CTX"]
        for k, eq in enumerate(early + late):
            if k:
                tokens.append("SEP")
            tokens += eq
        tokens += ["Q"] + q_vars + ["ANS"] * n + ["EOS"]
        targets = [f"D{vals[i][sides[i]]}" for i in order]
        probe = min(broken) if broken else rng.randrange(n)
        return Sample(
            self.name, difficulty, tokens, targets,
            epistemic=DERIVED, requires_distinction=int(condition),
            meta={"D": 3, "R": R, "L": len(tokens),
                  "pair": (f"X{2 * probe}", f"X{2 * probe + 1}"),
                  "vars": {f"X{2 * i + s}": vals[i][s]
                           for i in range(n) for s in (0, 1)}})

    def solve(self, sample: Sample) -> list[str]:
        import math
        toks = sample.tokens
        s1: dict[frozenset, int] = {}
        s2: dict[frozenset, int] = {}
        geq: dict[frozenset, tuple[int, int]] = {}
        for eq in _split_eqs(toks):
            xs = [int(t[1:]) for t in eq
                  if t.startswith("X") and t[1:].isdigit()]
            if not xs:
                continue                            # inert filler
            key = frozenset(xs)
            if "GEQ" in eq:
                geq[key] = (int(eq[0][1:]), int(eq[2][1:]))
            elif "POW" in eq:
                s2[key] = parse_num(eq[eq.index("EQ") + 1:])
            else:
                s1[key] = parse_num(eq[eq.index("EQ") + 1:])
        out: dict[int, int] = {}
        for key, s in s1.items():
            p2 = s2[key]
            p = (s * s - p2) // 2                   # elementary symmetric e2
            disc = s * s - 4 * p
            r = math.isqrt(disc)
            assert r * r == disc and (s + r) % 2 == 0
            hi, lo = geq[key]
            out[hi], out[lo] = (s + r) // 2, (s - r) // 2
        q0 = toks.index("Q") + 1
        return [f"D{out[int(t[1:])]}" for t in toks[q0: toks.index("ANS")]]


class RationalTrapGenerator(TaskGenerator):
    name = "rational_trap"
    family_token = "F1"
    min_difficulty = 1
    max_difficulty = 4

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        a = rng.randint(1, 4 + difficulty)
        trap = rng.random() < 0.5
        b = 2 * a if trap else rng.choice(
            [x for x in range(1, 19) if x != 2 * a and 0 <= x - a <= 9])
        cand = b - a                    # from simplifying to x + a = b
        tokens = [self.family_token, "BOS", "CTX", "LPAR", "X0", "POW", "D2",
                  "MINUS", *num_tokens(a * a), "RPAR", "DIV", "LPAR", "X0",
                  "MINUS", *num_tokens(a), "RPAR", "EQ", *num_tokens(b),
                  "SEP", "X0", "NEQ", *num_tokens(a),
                  "Q", "X0", "ANS", "EOS"]
        target = ["NO_SOLUTION"] if trap else [f"D{cand}"]
        return Sample(
            self.name, difficulty, tokens, target,
            epistemic=INCONSISTENT if trap else DERIVED,
            reconsider_label=int(trap),
            meta={"D": 3, "R": 1, "L": len(tokens),
                  "attractor": f"D{cand}" if 0 <= cand <= 9 else "D0",
                  "vars": {} if trap else {"X0": cand}})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        neq = toks.index("NEQ")
        excluded = parse_num(toks[neq + 1: toks.index("Q")])
        eq = toks.index("EQ")
        b = parse_num(toks[eq + 1: toks.index("SEP")])
        cand = b - excluded
        return ["NO_SOLUTION"] if cand == excluded else [f"D{cand}"]

    def score(self, sample: Sample, pred_tokens: list[str]) -> dict:
        correct = pred_tokens == sample.targets
        trap = sample.reconsider_label == 1
        pred = pred_tokens[0] if pred_tokens else "PAD"
        return {"correct": correct, "is_unknown": trap,
                "unsupported_assertion": trap and pred.startswith("D"),
                "over_abstention": (not trap) and pred == "NO_SOLUTION"}


class IterApplyGenerator(TaskGenerator):
    name = "iter_apply"
    family_token = "F2"
    min_difficulty = 1
    max_difficulty = 12                 # k applications = reasoning depth

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        k = difficulty
        c = rng.choice([2, 3])
        e = rng.randint(0, 9)
        x0 = rng.randint(0, 9)
        v = x0
        for _ in range(k):
            v = c * v + e
        w = len(num_tokens(v))
        w = max(w, 6)
        tokens = [self.family_token, "BOS", "CTX", "FN", "LPAR", "X0", "RPAR",
                  "EQ", *num_tokens(c), "TIMES", "X0", "PLUS", *num_tokens(e),
                  "SEP", "APPLY", *num_tokens(k), "FN", *num_tokens(x0),
                  "Q"] + ["ANS"] * w + ["EOS"]
        return Sample(self.name, k, tokens, num_tokens(v, width=w),
                      epistemic=DERIVED,
                      meta={"D": k, "R": 1, "L": len(tokens)})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        eq = toks.index("EQ")
        c = parse_num(toks[eq + 1: toks.index("TIMES")])
        e = parse_num(toks[toks.index("PLUS") + 1: toks.index("SEP")])
        ap = toks.index("APPLY")
        fn2 = toks.index("FN", ap)
        k = parse_num(toks[ap + 1: fn2])
        x = parse_num(toks[fn2 + 1: toks.index("Q")])
        for _ in range(k):
            x = c * x + e
        w = sum(1 for t in toks if t == "ANS")
        return num_tokens(x, width=w)

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        tgt = sample.targets
        p = (pred_tokens + ["PAD"] * len(tgt))[: len(tgt)]
        return float(sum(a != b for a, b in zip(p, tgt)))


class CongruenceGenerator(TaskGenerator):
    name = "congruence"
    family_token = "F3"
    min_difficulty = 1
    max_difficulty = 3                  # number of modular constraints

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        m_count = difficulty
        mods = rng.sample([2, 3, 4, 5, 6, 7], m_count)
        mode = rng.choices(["unique", "under", "inconsistent"],
                           weights=[0.45, 0.3, 0.25])[0]
        x = rng.randint(0, 9)
        residues = [x % m for m in mods]
        if mode == "inconsistent" and m_count >= 1:
            # perturb one residue so NO value in 0..9 satisfies all
            for _ in range(50):
                i = rng.randrange(m_count)
                residues[i] = (residues[i] + rng.randint(1, mods[i] - 1)) % mods[i]
                if not self._solutions(mods, residues):
                    break
                residues[i] = x % mods[i]
        sols = self._solutions(mods, residues)
        status = ("inconsistent" if not sols
                  else "unique" if len(sols) == 1 else "under")

        tokens = [self.family_token, "BOS", "CTX", "X0", "GEQ", "D0", "SEP",
                  "X0", "LEQ", "D9"]
        for m, r in zip(mods, residues):
            tokens += ["SEP", "X0", "MOD", *num_tokens(m), "EQ", *num_tokens(r)]
        tokens += ["Q", "X0", "ANS", "EOS"]
        target = ([f"D{sols[0]}"] if status == "unique" else
                  ["UNDERDETERMINED"] if status == "under" else ["NO_SOLUTION"])
        epi = (DERIVED if status == "unique" else
               UNDERDET if status == "under" else INCONSISTENT)
        return Sample(self.name, m_count, tokens, target, epistemic=epi,
                      meta={"D": m_count, "R": 1, "L": len(tokens),
                            "status": status,
                            "vars": {"X0": sols[0]} if status == "unique" else {}})

    @staticmethod
    def _solutions(mods, residues) -> list[int]:
        return [v for v in range(10)
                if all(v % m == r for m, r in zip(mods, residues))]

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        mods, residues = [], []
        for chunk in _split_eqs(toks):
            if "MOD" in chunk:
                mi = chunk.index("MOD")
                ei = chunk.index("EQ")
                mods.append(parse_num(chunk[mi + 1: ei]))
                residues.append(parse_num(chunk[ei + 1:]))
        sols = self._solutions(mods, residues)
        if not sols:
            return ["NO_SOLUTION"]
        return [f"D{sols[0]}"] if len(sols) == 1 else ["UNDERDETERMINED"]

    def score(self, sample: Sample, pred_tokens: list[str]) -> dict:
        correct = pred_tokens == sample.targets
        status = sample.meta["status"]
        is_unknown = status != "unique"
        pred = pred_tokens[0] if pred_tokens else "PAD"
        return {"correct": correct, "is_unknown": is_unknown, "status": status,
                "unsupported_assertion": is_unknown and pred.startswith("D"),
                "over_abstention": (not is_unknown)
                and pred in ("UNDERDETERMINED", "NO_SOLUTION")}


def _split_eqs(toks: list[str]) -> list[list[str]]:
    span = toks[3: toks.index("Q")]
    out, cur = [], []
    for t in span + ["SEP"]:
        if t == "SEP":
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(t)
    return out
