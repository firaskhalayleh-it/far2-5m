"""M6 rewrite — symbolic equivalence under the ring axioms (distribution,
commutation, combination, identities). Two prefix expression trees; answer
YES iff they denote the same polynomial. The YES variant is produced by
semantics-preserving transformations (commute, distribute, identity padding);
the NO variant perturbs one leaf or operator and is verified inequivalent by
the exact polynomial normal form — which is also the independent solver.

difficulty = depth of the first expression tree."""

import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.poly import p_max_coef, p_max_deg, poly_of_prefix

Tree = list | str


def _build(rng: random.Random, depth: int) -> Tree:
    if depth == 0:
        return (f"X{rng.randrange(3)}" if rng.random() < 0.5
                else f"D{rng.randint(0, 9)}")
    deep = _build(rng, depth - 1)
    shallow = _build(rng, rng.randint(0, min(depth - 1, 2)))
    l, r = (deep, shallow) if rng.random() < 0.5 else (shallow, deep)
    op = "TIMES" if rng.random() < 0.35 else rng.choice(["PLUS", "MINUS"])
    return [op, l, r]


def _flat(t: Tree) -> list[str]:
    return [t] if isinstance(t, str) else [t[0]] + _flat(t[1]) + _flat(t[2])


def _poly(t: Tree) -> dict:
    return poly_of_prefix(_flat(t), [0])


def _transform(rng: random.Random, t: Tree) -> Tree:
    """Semantics-preserving rewrite: recursive commutes + one distribution +
    occasional identity wrap (TIMES 1 / PLUS 0)."""
    def go(node: Tree) -> Tree:
        if isinstance(node, str):
            return node
        op, a, b = node[0], go(node[1]), go(node[2])
        if op in ("PLUS", "TIMES") and rng.random() < 0.5:
            a, b = b, a
        if (op == "TIMES" and isinstance(b, list) and b[0] == "PLUS"
                and rng.random() < 0.7):
            return ["PLUS", ["TIMES", a, b[1]], ["TIMES", a, b[2]]]
        return [op, a, b]

    out = go(t)
    if rng.random() < 0.4:
        out = (["TIMES", "D1", out] if rng.random() < 0.5
               else ["PLUS", "D0", out])
    return out


def _perturb(rng: random.Random, t: Tree) -> Tree:
    """One local edit; caller verifies the polynomial actually changed."""
    flat_positions: list[list[int]] = []

    def walk(node, path):
        if isinstance(node, str):
            flat_positions.append(path)
        else:
            walk(node[1], path + [1])
            walk(node[2], path + [2])

    walk(t, [])
    path = rng.choice(flat_positions)
    import copy
    out = copy.deepcopy(t)
    ref, leaf_path = out, path
    for p in leaf_path[:-1]:
        ref = ref[p]
    leaf = ref[leaf_path[-1]] if leaf_path else out
    new = (f"D{rng.randint(0, 9)}" if rng.random() < 0.5
           else f"X{rng.randrange(3)}")
    if not leaf_path:
        return new
    ref[leaf_path[-1]] = new
    return out


class RewriteGenerator(TaskGenerator):
    name = "rewrite"
    family_token = "F6"
    min_difficulty = 2
    max_difficulty = 12

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        for _ in range(80):
            a = _build(rng, difficulty)
            pa = _poly(a)
            if p_max_deg(pa) <= 6 and p_max_coef(pa) <= 5000:
                break
        equivalent = rng.random() < 0.5
        for _ in range(80):
            b = _transform(rng, a) if equivalent else _perturb(rng, a)
            if (_poly(b) == pa) == equivalent:
                break
        else:
            b, equivalent = _transform(rng, a), True

        tokens = ([self.family_token, "BOS", "CTX"] + _flat(a) + ["EQUIV"]
                  + _flat(b) + ["Q", "ANS", "EOS"])
        return Sample(self.name, difficulty, tokens,
                      ["YES" if equivalent else "NO"],
                      epistemic=DERIVED,
                      meta={"D": difficulty, "R": 1, "L": len(tokens)})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        span = toks[3: toks.index("Q")]
        cut = span.index("EQUIV")
        pa = poly_of_prefix(span[:cut], [0])
        pb = poly_of_prefix(span[cut + 1:], [0])
        return ["YES" if pa == pb else "NO"]
