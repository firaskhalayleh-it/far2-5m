"""M3 expr_eval — exact evaluation of prefix expression trees with TREE DEPTH
controlled independently of token count. Two independent dials:

  depth (the difficulty)  exact tree depth — the deep child recurses, the
                          shallow child stays depth <= 2 (caterpillar spine)
  leaf width              leaves are 1..4-digit numbers; multi-digit leaves
                          are LPAR..RPAR-wrapped so prefix stays unambiguous

so a depth-2 tree over wide leaves is LONGER than a depth-6 tree over narrow
ones — token count cannot proxy for depth. TIMES is allowed only when the
exact product stays in range; |value| <= 9999 by resampling. Answers are a
fixed 5-slot format: sign token (POS/NEG) + 4 zero-padded digits.

difficulty = exact tree depth D."""

import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import num_tokens, parse_num

OPS = {"PLUS": lambda a, b: a + b, "MINUS": lambda a, b: a - b,
       "TIMES": lambda a, b: a * b}
# leaf digit-width distribution: narrow leaves keep deep trees short, wide
# leaves make shallow trees long — the decorrelation lives here
WIDTHS = [1, 1, 1, 1, 2, 2, 3, 3, 4]


def _leaf(rng: random.Random) -> tuple[list[str], int]:
    k = rng.choice(WIDTHS)
    v = rng.randint(0, 10 ** k - 1)
    if k == 1:
        return [f"D{v}"], v
    return ["LPAR"] + num_tokens(v, width=k) + ["RPAR"], v


def _build(rng: random.Random, depth: int) -> tuple[list[str], int]:
    if depth == 0:
        return _leaf(rng)
    deep, dv = _build(rng, depth - 1)
    # the shallow child is often a bare single digit so DEEP trees can be
    # SHORT (thin caterpillar); otherwise a small subtree so SHALLOW trees
    # can be LONG — depth and token count decouple
    if rng.random() < 0.5:
        v = rng.randint(0, 9)
        shallow, sv = [f"D{v}"], v
    else:
        shallow, sv = _build(rng, rng.randint(0, min(depth - 1, 2)))
    left_deep = rng.random() < 0.5
    lt, lv, rt, rv = ((deep, dv, shallow, sv) if left_deep
                      else (shallow, sv, deep, dv))
    ops = ["PLUS", "MINUS"]
    if abs(lv * rv) <= 9999 and rng.random() < 0.3:
        op = "TIMES"
    else:
        op = rng.choice(ops)
    return [op] + lt + rt, OPS[op](lv, rv)


class ExprEvalGenerator(TaskGenerator):
    name = "expr_eval"
    family_token = "F3"
    min_difficulty = 2
    max_difficulty = 16

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        for _ in range(60):
            expr, v = _build(rng, difficulty)
            if abs(v) <= 9999:
                break
        else:                                   # PLUS-only caterpillar fallback
            expr = [f"D{rng.randint(0, 9)}"]
            for _ in range(difficulty):
                expr = ["PLUS", f"D{rng.randint(0, 9)}"] + expr
            v = _eval_prefix(expr, [0])[0]
        tokens = ([self.family_token, "BOS", "CTX"] + expr
                  + ["Q"] + ["ANS"] * 5 + ["EOS"])
        sign = "NEG" if v < 0 else "POS"
        targets = [sign] + num_tokens(abs(v), width=4)
        return Sample(self.name, difficulty, tokens, targets,
                      epistemic=DERIVED,
                      meta={"D": difficulty, "R": 1, "L": len(tokens)})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        expr = toks[3: toks.index("Q")]
        v = _eval_prefix(expr, [0])[0]
        return (["NEG" if v < 0 else "POS"] + num_tokens(abs(v), width=4))

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        tgt = sample.targets
        p = (pred_tokens + ["PAD"] * len(tgt))[: len(tgt)]
        return float(sum(a != b for a, b in zip(p, tgt)))


def _eval_prefix(expr: list[str], pos: list[int]) -> tuple[int, None]:
    """Recursive-descent exact evaluation; pos is a 1-cell cursor. Leaves are
    single digit tokens or LPAR <digit run> RPAR."""
    t = expr[pos[0]]
    pos[0] += 1
    if t in OPS:
        a, _ = _eval_prefix(expr, pos)
        b, _ = _eval_prefix(expr, pos)
        return OPS[t](a, b), None
    if t == "LPAR":
        j = pos[0]
        while expr[j] != "RPAR":
            j += 1
        v = parse_num(expr[pos[0]: j])
        pos[0] = j + 1
        return v, None
    return parse_num([t]), None
