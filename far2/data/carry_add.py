"""M0 carry_add — multi-digit addition/subtraction with INDEPENDENT control of
surface length L (digit count) and reasoning depth D (exact carry/borrow-chain
length). `99999999 + 1` is deep; `10000002 + 10000003` is long but shallow.
The matched counterexample is built in: any (L, D) cell can be generated, so
"long input -> think more" is separable from "deep carry -> think more".

difficulty = D (carry depth). Digit count defaults to a train-range draw;
eval passes n_digits explicitly for the extrapolation matrix."""

import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import digit_tok, num_tokens, parse_num


class CarryAddGenerator(TaskGenerator):
    name = "carry_add"
    family_token = "F0"
    min_difficulty = 0                 # D = 0: no carries at all
    max_difficulty = 32

    def sample(self, rng: random.Random, difficulty: int,
               n_digits: int | None = None) -> Sample:
        d = difficulty
        n = n_digits or rng.randint(max(d + 1, 3), max(d + 1, 8))
        n = max(n, d + 1)
        o = rng.randint(0, n - d) if d else 0

        a, b = [0] * n, [0] * n            # LSD at index 0
        for i in range(n):
            top = i == n - 1
            if d and o <= i < o + d:
                if i == o:                  # chain start: digit sum >= 10
                    s = rng.randint(10, 18)
                    lo = max(s - 9, 1)
                    a[i] = rng.randint(lo, 9)
                else:                       # chain body: sum exactly 9
                    a[i] = rng.randint(1, 8) if top else rng.randint(0, 9)
                    s = 9
                b[i] = s - a[i]
            else:                           # outside the chain: sum <= 8
                s = rng.randint(2, 8) if top else rng.randint(0, 8)
                a[i] = rng.randint(1, s - 1) if top else rng.randint(0, s)
                b[i] = s - a[i]

        av = int("".join(str(x) for x in reversed(a)))
        bv = int("".join(str(x) for x in reversed(b)))
        total = av + bv
        w = n + 1
        if rng.random() < 0.5:
            lhs, rhs, ans, op = av, bv, total, "PLUS"
        else:                               # (a+b) - a = b: borrow chain = d
            lhs, rhs, ans, op = total, av, bv, "MINUS"

        tokens = [self.family_token, "BOS", "CTX", *num_tokens(lhs), op,
                  *num_tokens(rhs), "Q"] + ["ANS"] * w + ["EOS"]
        return Sample(self.name, d, tokens, num_tokens(ans, width=w),
                      epistemic=DERIVED,
                      meta={"D": d, "R": 1, "L": len(tokens), "op": op})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        op = "PLUS" if "PLUS" in toks else "MINUS"
        i = toks.index(op)
        j = toks.index("Q")
        lhs = parse_num(toks[3:i])
        rhs = parse_num(toks[i + 1:j])
        w = sum(1 for t in toks if t == "ANS")
        return num_tokens(lhs + rhs if op == "PLUS" else lhs - rhs, width=w)

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        """Wrong digits counted from the least-significant side."""
        tgt = sample.targets
        p = (pred_tokens + ["PAD"] * len(tgt))[: len(tgt)]
        return float(sum(a != b for a, b in zip(p, tgt)))


class MultiplyGenerator(TaskGenerator):
    """M1 — exact multi-digit multiplication. difficulty = digits of the first
    operand; the second draws 1..difficulty digits. D = partial-product count
    (digits of b); answer width = len(a)+len(b), zero-padded."""

    name = "multiply"
    family_token = "F1"
    min_difficulty = 1
    max_difficulty = 8

    def sample(self, rng: random.Random, difficulty: int,
               b_digits: int | None = None) -> Sample:
        na = difficulty
        nb = b_digits or rng.randint(1, difficulty)
        a = rng.randint(10 ** (na - 1), 10 ** na - 1) if na > 1 else rng.randint(1, 9)
        b = rng.randint(10 ** (nb - 1), 10 ** nb - 1) if nb > 1 else rng.randint(1, 9)
        w = na + nb
        tokens = [self.family_token, "BOS", "CTX", *num_tokens(a), "TIMES",
                  *num_tokens(b), "Q"] + ["ANS"] * w + ["EOS"]
        return Sample(self.name, difficulty, tokens, num_tokens(a * b, width=w),
                      epistemic=DERIVED,
                      meta={"D": nb, "R": 1, "L": len(tokens)})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        i, j = toks.index("TIMES"), toks.index("Q")
        a, b = parse_num(toks[3:i]), parse_num(toks[i + 1:j])
        w = sum(1 for t in toks if t == "ANS")
        return num_tokens(a * b, width=w)

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        tgt = sample.targets
        p = (pred_tokens + ["PAD"] * len(tgt))[: len(tgt)]
        return float(sum(a != b for a, b in zip(p, tgt)))


class BaseArithGenerator(TaskGenerator):
    """M2 — counterfactual bases: the base is DECLARED IN THE PROBLEM and the
    model must execute addition under it. difficulty = the base itself
    (train {5,7,10} via Mix choices; test {6,9,11}) — reasoning vs recitation.
    DA is the digit for value 10 (base 11)."""

    name = "base_arith"
    family_token = "F2"
    min_difficulty = 5
    max_difficulty = 11

    def sample(self, rng: random.Random, difficulty: int,
               n_digits: int | None = None) -> Sample:
        base = difficulty
        n = n_digits or rng.randint(2, 5)
        a = [rng.randrange(base) for _ in range(n)]
        b = [rng.randrange(base) for _ in range(n)]
        a[-1] = max(a[-1], 1)
        b[-1] = max(b[-1], 1)
        av = sum(x * base ** i for i, x in enumerate(a))
        bv = sum(x * base ** i for i, x in enumerate(b))
        w = n + 1
        carry_depth = self._carry_depth(a, b, base)
        tokens = [self.family_token, "BOS", "CTX", "BASE", *num_tokens(base),
                  "SEP", *num_tokens(av, base=base), "PLUS",
                  *num_tokens(bv, base=base), "Q"] + ["ANS"] * w + ["EOS"]
        return Sample(self.name, base, tokens,
                      num_tokens(av + bv, width=w, base=base),
                      epistemic=DERIVED,
                      meta={"D": carry_depth, "R": 1, "L": len(tokens),
                            "base": base})

    @staticmethod
    def _carry_depth(a, b, base) -> int:
        carry, run, best = 0, 0, 0
        for x, y in zip(a, b):
            carry = 1 if x + y + carry >= base else 0
            run = run + 1 if carry else 0
            best = max(best, run)
        return best

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        s = toks.index("SEP")
        base = parse_num(toks[4:s])
        i, j = toks.index("PLUS"), toks.index("Q")
        a = parse_num(toks[s + 1:i], base=base)
        b = parse_num(toks[i + 1:j], base=base)
        w = sum(1 for t in toks if t == "ANS")
        return num_tokens(a + b, width=w, base=base)
