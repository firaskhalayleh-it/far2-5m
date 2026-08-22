"""M8 extraneous — the mathematical RECONSIDER trap. sqrt(x + a) = x - b:
squaring yields a quadratic with two candidates r and e; a candidate is a
true root only if x - b >= 0. The naive first candidate (the smaller one) is
sometimes extraneous — a locally plausible trajectory invalidated by the
original constraint. Task convention (learned from the grammar): answer the
SMALLEST VALID root.

reconsider_label = 1 iff the smaller candidate is extraneous (trap);
matched controls have both candidates valid (identical grammar and length).
meta["attractor"] is the tempting candidate's digit token, which powers the
recovery metric via the per-iteration answer probe."""

import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import num_tokens, parse_num


class ExtraneousGenerator(TaskGenerator):
    name = "extraneous"
    family_token = "F8"
    min_difficulty = 2
    max_difficulty = 8

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        trap = rng.random() < 0.5
        b = min(difficulty, 7 if trap else 8)
        if trap:                       # smaller candidate e < b: extraneous
            e = rng.randint(max(0, 2 * b - 8), b - 1)   # keeps r = 2b+1-e <= 9
        else:                          # both valid: e in {b, b+1}, r = 2b+1-e
            e = rng.choice([b, b + 1])
        r = 2 * b + 1 - e
        assert 0 <= r <= 9 and 0 <= e <= 9
        lo, hi = min(e, r), max(e, r)
        a = b * b - e * r              # from x^2 -(2b+1)x + (b^2 - a) = 0

        tokens = [self.family_token, "BOS", "CTX", "SQRT", "LPAR", "X0",
                  "PLUS"]
        tokens += (["NEG"] if a < 0 else []) + num_tokens(abs(a))
        tokens += ["RPAR", "EQ", "X0", "MINUS", *num_tokens(b),
                   "Q", "X0", "ANS", "EOS"]
        valid = [c for c in (lo, hi) if c >= b]
        answer = min(valid)
        return Sample(
            self.name, difficulty, tokens, [f"D{answer}"],
            epistemic=DERIVED,
            reconsider_label=int(lo < b),
            meta={"D": 3, "R": 1, "L": len(tokens),
                  "attractor": f"D{lo}", "candidates": (lo, hi),
                  "vars": {"X0": answer}})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        j = toks.index("RPAR")
        a = parse_num(toks[toks.index("PLUS") + 1: j])
        k = toks.index("MINUS")
        b = parse_num(toks[k + 1: toks.index("Q")])
        cands = [c for c in range(10)
                 if c * c - (2 * b + 1) * c + (b * b - a) == 0]
        valid = [c for c in cands if c >= b]
        return [f"D{min(valid)}"]

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        v = parse_num(pred_tokens[:1]) if pred_tokens else None
        t = parse_num(sample.targets[:1])
        return abs(v - t) if v is not None else 10.0
