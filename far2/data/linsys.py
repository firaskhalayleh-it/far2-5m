"""M4 linsys — integer linear systems with MATHEMATICAL epistemic semantics.
The queried variable is DERIVED (unique value), the system is INCONSISTENT
(correct output: NO_SOLUTION), or the query is UNDERDETERMINED (correct
output: UNDERDETERMINED, never a guess). Exact rational Gaussian elimination
(fractions.Fraction) is both the constructor's check and the independent
solver — there is provably no justified numeric answer in the non-unique
cases, which makes this the honest hallucination test.

difficulty = number of variables n. All answers are ONE slot (digit /
UNDERDETERMINED / NO_SOLUTION), so slot count never leaks the status."""

import random
from fractions import Fraction

from far2.config import DERIVED, INCONSISTENT, OBSERVED, UNDERDET
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import num_tokens, parse_num


def rref(rows: list[tuple[list[int], int]]):
    """Exact reduced row echelon form. rows: [(coefs, rhs)].
    Returns (matrix, pivot_columns, inconsistent)."""
    m = [[Fraction(c) for c in r] + [Fraction(v)] for r, v in rows]
    if not m:
        return [], [], False
    n = len(m[0]) - 1
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        m[r] = [x / m[r][c] for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                m[i] = [a - m[i][c] * b for a, b in zip(m[i], m[r])]
        piv.append(c)
        r += 1
        if r == len(m):
            break
    incons = any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in m)
    return m, piv, incons


def var_status(rows, q: int, n: int) -> tuple[str, Fraction | None]:
    """('inconsistent'|'unique'|'under', value-if-unique) for variable q."""
    m, piv, incons = rref(rows)
    if incons:
        return "inconsistent", None
    if q in piv:
        row = m[piv.index(q)]
        free = [c for c in range(n) if c not in piv]
        if all(row[c] == 0 for c in free):
            return "unique", row[-1]
    return "under", None


def render_equation(coefs: list[int], rhs: int, vars_: list[int]) -> list[str]:
    """c0*Xi + c1*Xj ... EQ rhs (coefficient always explicit; NEG prefixes)."""
    out: list[str] = []
    for k, (c, vi) in enumerate(zip(coefs, vars_)):
        if k:
            out.append("PLUS")
        if c < 0:
            out.append("NEG")
        out += num_tokens(abs(c)) + ["TIMES", f"X{vi}"]
    return out + ["EQ"] + num_tokens(rhs)


def parse_equations(tokens: list[str], n_vars: int):
    """Inverse of render_equation over the CTX..Q span; SEP-separated."""
    span = tokens[3: tokens.index("Q")]
    rows, eq = [], []
    for t in span + ["SEP"]:
        if t == "SEP":
            if eq:
                rows.append(_parse_eq(eq, n_vars))
                eq = []
        else:
            eq.append(t)
    return rows


def _parse_eq(eq: list[str], n_vars: int):
    coefs = [0] * n_vars
    i, sign, cur = 0, 1, 0
    lhs, rhs_toks = eq[: eq.index("EQ")], eq[eq.index("EQ") + 1:]
    while i < len(lhs):
        t = lhs[i]
        if t == "PLUS":
            sign = 1
        elif t == "NEG":
            sign = -1
        elif t.startswith("D"):
            cur = cur * 10 + parse_num([t])
        elif t == "TIMES":
            pass
        elif t.startswith("X"):
            coefs[int(t[1:])] += sign * cur
            sign, cur = 1, 0
        i += 1
    return coefs, parse_num(rhs_toks)


class LinSysGenerator(TaskGenerator):
    name = "linsys"
    family_token = "F4"
    min_difficulty = 2
    max_difficulty = 8

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        n = difficulty
        x = [rng.randint(0, 9) for _ in range(n)]
        mode = rng.choices(["unique", "under", "inconsistent"],
                           weights=[0.5, 0.25, 0.25])[0]
        rows, epi = self._construct(rng, n, x, mode)
        q = self._pick_query(rng, rows, n, mode)
        status, val = var_status(rows, q, n)

        tokens = [self.family_token, "BOS", "CTX"]
        order = list(range(len(rows)))
        rng.shuffle(order)
        for k, ri in enumerate(order):
            if k:
                tokens.append("SEP")
            tokens += render_equation(*self._split(rows[ri]))
        tokens += ["Q", f"X{q}", "ANS", "EOS"]

        if status == "unique":
            target = [f"D{int(val)}"]
        elif status == "inconsistent":
            target, epi = ["NO_SOLUTION"], INCONSISTENT
        else:
            target, epi = ["UNDERDETERMINED"], UNDERDET
        meta = {"D": n, "R": len(rref(rows)[1]), "L": len(tokens),
                "status": status, "query": q,
                "vars": {f"X{i}": x[i] for i in range(n)}
                if status != "inconsistent" else {}}
        return Sample(self.name, n, tokens, target, epistemic=epi, meta=meta)

    @staticmethod
    def _split(row):
        coefs, rhs = row
        vars_ = [i for i, c in enumerate(coefs) if c]
        return [coefs[i] for i in vars_], rhs, vars_

    def _construct(self, rng, n, x, mode):
        def rand_row(vs=None):
            vs = vs or rng.sample(range(n), rng.randint(2, min(3, n)))
            coefs = [0] * n
            for v in vs:
                coefs[v] = rng.choice([-3, -2, -1, 1, 2, 3])
            return coefs, sum(c * x[i] for i, c in enumerate(coefs))

        epi = DERIVED
        rows = [rand_row() for _ in range(n)]
        if mode == "unique":
            for i in range(n):                    # guarantee full rank
                if var_status(rows, i, n)[0] != "unique":
                    unit = [0] * n
                    unit[i] = 1
                    rows.append((unit, x[i]))
            if rng.random() < 0.2:                # direct read-off case
                q0 = rng.randrange(n)
                unit = [0] * n
                unit[q0] = 1
                rows.append((unit, x[q0]))
                epi = OBSERVED
        elif mode == "under":
            drop = rng.randrange(n)               # leave a variable loose
            rows = [r for r in rows if r[0][drop] == 0]
            rows = rows or [rand_row([v for v in range(n) if v != drop]
                                     if n > 2 else [1 - drop])]
        else:                                     # inconsistent
            c0, r0 = rows[0]
            rows.append((list(c0), r0 + rng.randint(1, 5)))
        return rows, epi

    def _pick_query(self, rng, rows, n, mode):
        if mode == "under":
            free = [q for q in range(n) if var_status(rows, q, n)[0] == "under"]
            if free:
                return rng.choice(free)
        return rng.randrange(n)

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        n = sample.difficulty
        q = int(toks[toks.index("Q") + 1][1:])
        rows = parse_equations(toks, n)
        status, val = var_status(rows, q, n)
        if status == "unique":
            return [f"D{int(val)}"]
        return ["NO_SOLUTION"] if status == "inconsistent" else ["UNDERDETERMINED"]

    def score(self, sample: Sample, pred_tokens: list[str]) -> dict:
        correct = pred_tokens == sample.targets
        status = sample.meta["status"]
        is_unknown = status != "unique"
        pred = pred_tokens[0] if pred_tokens else "PAD"
        return {"correct": correct, "is_unknown": is_unknown, "status": status,
                "unsupported_assertion": is_unknown and pred.startswith("D"),
                "over_abstention": (not is_unknown)
                and pred in ("UNDERDETERMINED", "NO_SOLUTION")}

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float:
        if sample.meta["status"] != "unique" or not pred_tokens:
            return float(pred_tokens != sample.targets)
        v = parse_num(pred_tokens[:1])
        t = parse_num(sample.targets[:1])
        return abs(v - t) if v is not None and t is not None else 10.0
