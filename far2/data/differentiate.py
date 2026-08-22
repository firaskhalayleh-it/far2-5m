"""M7 differentiate — formal symbolic differentiation of single-variable
polynomials rendered as prefix trees. The RENDERED DEPTH is controlled
independently of the semantic content: x^e appears either as POW X0 De
(depth 1) or as an explicit nested product chain (depth e, forcing the
product rule recursively), and semantically neutral wrappers (TIMES D1 ... /
PLUS D0 ...) pad depth exactly. Same polynomial, different reasoning depth —
the compositional-recursion probe the surface metrics need.

Answer format (fixed 12 slots, no length leak): three monomial slots in
descending exponent, each = 3-digit coefficient + 1-digit exponent; absent
monomials are all-zero. difficulty = exact rendered tree depth D."""

import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.poly import poly_of_prefix
from far2.data.vocab import num_tokens

Tree = list | str


def _depth(t: Tree) -> int:
    if isinstance(t, str):
        return 0
    if isinstance(t, tuple):                     # ("POW", e) node
        return 1
    return 1 + max(_depth(t[1]), _depth(t[2]))


def _pow_chain(exp: int) -> Tree:
    t: Tree = "X0"
    for _ in range(exp - 1):
        t = ["TIMES", "X0", t]
    return t


def _monomial(rng: random.Random, coef: int, exp: int, budget: int) -> Tree:
    if exp == 0:
        return f"D{coef}"
    body: Tree = (("POW", exp) if exp > budget or rng.random() < 0.35
                  else _pow_chain(exp))
    if coef != 1 or rng.random() < 0.5:
        return ["TIMES", f"D{coef}", body]
    return body


def _flat(t: Tree) -> list[str]:
    if isinstance(t, str):
        return [t]
    if isinstance(t, tuple):                     # ("POW", e)
        return ["POW", "X0", f"D{t[1]}"]
    return [t[0]] + _flat(t[1]) + _flat(t[2])


class DifferentiateGenerator(TaskGenerator):
    name = "differentiate"
    family_token = "F7"
    min_difficulty = 2
    max_difficulty = 16

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        d = difficulty
        span = min(9, max(2, d))
        k = rng.randint(1, min(3, span))
        exps = rng.sample(range(span), k)
        coefs = {e: rng.randint(1, 9) for e in exps}

        parts = [_monomial(rng, coefs[e], e, budget=d - 1)
                 for e in sorted(exps, reverse=True)]
        tree: Tree = parts[0]
        for p in parts[1:]:
            tree = ["PLUS", tree, p] if rng.random() < 0.5 else ["PLUS", p, tree]
        while _depth(tree) < d:                  # exact-depth neutral padding
            tree = (["TIMES", "D1", tree] if rng.random() < 0.5
                    else ["PLUS", "D0", tree])

        tokens = ([self.family_token, "BOS", "CTX", "DERIV"] + _flat(tree)
                  + ["Q"] + ["ANS"] * 12 + ["EOS"])
        targets = _derivative_tokens({e: c for e, c in coefs.items()})
        return Sample(self.name, d, tokens, targets, epistemic=DERIVED,
                      meta={"D": _depth(tree), "R": 1, "L": len(tokens),
                            "poly": dict(coefs)})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        span = toks[toks.index("DERIV") + 1: toks.index("Q")]
        p = poly_of_prefix(span, [0])
        mono = {m[0]: c for m, c in p.items()}
        return _derivative_tokens(mono)

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        tgt = sample.targets
        p = (pred_tokens + ["PAD"] * len(tgt))[: len(tgt)]
        return float(sum(a != b for a, b in zip(p, tgt)))


def _derivative_tokens(mono: dict[int, int]) -> list[str]:
    """d/dx of {exp: coef} -> 3 canonical slots x (3-digit coef, 1-digit exp)."""
    deriv = {e - 1: e * c for e, c in mono.items() if e >= 1 and e * c != 0}
    out: list[str] = []
    for e in sorted(deriv, reverse=True)[:3]:
        out += num_tokens(deriv[e], width=3) + num_tokens(e, width=1)
    while len(out) < 12:
        out += num_tokens(0, width=3) + num_tokens(0, width=1)
    return out
