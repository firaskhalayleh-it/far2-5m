"""M9 optimize — constrained integer optimization with a PROTECTED GOAL.
Minimize c1*X0 + c2*X1 subject to X0 >= p, X1 >= q, and coupling constraints
X0 + X1 >= s_j. The tempting unconstrained answer (c1*p + c2*q) is wrong
whenever a coupling constraint binds; the goal span (objective + constraints)
must be preserved through reasoning, not overwritten by the greedy candidate.

difficulty = number of constraints. Answer = the exact optimal VALUE
(3 zero-padded digits). score() types the violation: a predicted value below
the true optimum claims an infeasible solution."""

import random

from far2.config import DERIVED
from far2.data.base import Sample, TaskGenerator
from far2.data.vocab import num_tokens, parse_num


class OptimizeGenerator(TaskGenerator):
    name = "optimize"
    family_token = "F9"
    min_difficulty = 2
    max_difficulty = 8

    def sample(self, rng: random.Random, difficulty: int) -> Sample:
        m = difficulty
        c1, c2 = rng.randint(1, 3), rng.randint(1, 3)
        p, q = rng.randint(0, 9), rng.randint(0, 9)
        n_couple = max(m - 2, 0)
        couples = sorted(rng.randint(p + q, p + q + 15) for _ in range(n_couple))

        tokens = [self.family_token, "BOS", "GOAL", "MIN",
                  *num_tokens(c1), "TIMES", "X0", "PLUS",
                  *num_tokens(c2), "TIMES", "X1", "SUBJ",
                  "X0", "GEQ", *num_tokens(p), "SEP",
                  "X1", "GEQ", *num_tokens(q)]
        for s in couples:
            tokens += ["SEP", "X0", "PLUS", "X1", "GEQ", *num_tokens(s)]
        tokens += ["Q"] + ["ANS"] * 3 + ["EOS"]

        val, (x0, x1) = _optimum(c1, c2, p, q, couples)
        return Sample(self.name, m, tokens, num_tokens(val, width=3),
                      epistemic=DERIVED,
                      meta={"D": m, "R": 2, "L": len(tokens),
                            "optimum": val, "greedy": c1 * p + c2 * q,
                            # collate keeps only 0..9 values for binding targets
                            "vars": {"X0": x0, "X1": x1}})

    def solve(self, sample: Sample) -> list[str]:
        toks = sample.tokens
        subj = toks.index("SUBJ")
        goal = toks[4: subj]
        t = goal.index("TIMES")
        c1 = parse_num(goal[:t])
        rest = goal[t + 2:]                    # after TIMES X0
        c2 = parse_num(rest[rest.index("PLUS") + 1: rest.index("TIMES")])
        p = q = 0
        couples = []
        for chunk in _chunks(toks[subj + 1: toks.index("Q")]):
            if chunk[:2] == ["X0", "GEQ"]:
                p = parse_num(chunk[2:])
            elif chunk[:2] == ["X1", "GEQ"]:
                q = parse_num(chunk[2:])
            else:                              # X0 PLUS X1 GEQ s
                couples.append(parse_num(chunk[4:]))
        val, _ = _optimum(c1, c2, p, q, couples)
        return num_tokens(val, width=3)

    def score(self, sample: Sample, pred_tokens: list[str]) -> dict:
        correct = pred_tokens == sample.targets
        v = parse_num(pred_tokens) if pred_tokens else None
        violation = ("infeasible_value" if v is not None
                     and v < sample.meta["optimum"] else None)
        return {"correct": correct, "goal_violation": violation,
                "claimed_greedy": v == sample.meta["greedy"]
                and sample.meta["greedy"] < sample.meta["optimum"]}


def _optimum(c1, c2, p, q, couples) -> tuple[int, tuple[int, int]]:
    """Exact: start at (p, q); satisfy the tightest coupling constraint by
    growing the cheaper variable."""
    S = max(couples) if couples else 0
    x0, x1 = p, q
    if x0 + x1 < S:
        slack = S - x0 - x1
        if c1 <= c2:
            x0 += slack
        else:
            x1 += slack
    return c1 * x0 + c2 * x1, (x0, x1)


def _chunks(tokens: list[str]):
    cur: list[str] = []
    for t in tokens + ["SEP"]:
        if t == "SEP":
            if cur:
                yield cur
                cur = []
        else:
            cur.append(t)
