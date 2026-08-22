"""Exact multivariate polynomial arithmetic over X0..X2 — the shared semantic
ground truth for the symbolic families (M6 rewrite, M7 differentiate). A
polynomial is a dict {(e0, e1, e2): coefficient}; zero coefficients are
dropped, so equality of dicts IS mathematical equivalence."""

from far2.data.vocab import parse_num

N_POLY_VARS = 3


def p_const(c: int) -> dict:
    return {(0, 0, 0): c} if c else {}


def p_var(i: int, exp: int = 1) -> dict:
    e = [0, 0, 0]
    e[i] = exp
    return {tuple(e): 1}


def p_add(p: dict, q: dict, sign: int = 1) -> dict:
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + sign * c
        if out[m] == 0:
            del out[m]
    return out


def p_mul(p: dict, q: dict) -> dict:
    out: dict = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(a + b for a, b in zip(m1, m2))
            out[m] = out.get(m, 0) + c1 * c2
            if out[m] == 0:
                del out[m]
    return out


def p_max_deg(p: dict) -> int:
    return max((sum(m) for m in p), default=0)


def p_max_coef(p: dict) -> int:
    return max((abs(c) for c in p.values()), default=0)


# ---- prefix-token parsing (shared by generators' independent solvers) ----
_OPS = {"PLUS": 1, "MINUS": -1, "TIMES": 0}


def poly_of_prefix(tokens: list[str], pos: list[int]) -> dict:
    """Parse a prefix expression into its exact polynomial. Grammar:
    PLUS/MINUS/TIMES <expr> <expr> | POW Xi Dn | Xi | <digit run>."""
    t = tokens[pos[0]]
    pos[0] += 1
    if t in ("PLUS", "MINUS"):
        a = poly_of_prefix(tokens, pos)
        b = poly_of_prefix(tokens, pos)
        return p_add(a, b, 1 if t == "PLUS" else -1)
    if t == "TIMES":
        a = poly_of_prefix(tokens, pos)
        b = poly_of_prefix(tokens, pos)
        return p_mul(a, b)
    if t == "POW":
        vi = int(tokens[pos[0]][1:])
        pos[0] += 1
        exp = parse_num([tokens[pos[0]]])
        pos[0] += 1
        return p_var(vi, exp) if exp else p_const(1)
    if t.startswith("X"):
        return p_var(int(t[1:]))
    # constants are SINGLE digit tokens inside expression trees — multi-digit
    # runs would be ambiguous in prefix notation when operands are adjacent
    return p_const(parse_num([t]))
