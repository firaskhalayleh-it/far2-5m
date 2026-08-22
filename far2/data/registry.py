"""Family registry and the weighted mixture sampler used by training.
Run `python -m far2.data.registry --show-samples` to eyeball real output.

GENERATORS are trainable mathematical families (M0-M9). TRANSFER_GENERATORS
are the held-out hidden benchmark (same cognitive conditions, unseen surface
grammar) — the sampler refuses to train on them; only the harness may touch
them. v0.4: the sampler also draws a random Abacus significance offset per
batch so all digit-significance embeddings train (see data/base.py)."""

import random
from dataclasses import dataclass

from far2.data.base import (FAMILY_ORDER, Sample, TaskGenerator,
                            randomize_family_token)
from far2.data.carry_add import (BaseArithGenerator, CarryAddGenerator,
                                 MultiplyGenerator)
from far2.data.differentiate import DifferentiateGenerator
from far2.data.expr_eval import ExprEvalGenerator
from far2.data.extraneous import ExtraneousGenerator
from far2.data.linsys import LinSysGenerator
from far2.data.optimize import OptimizeGenerator
from far2.data.rewrite import RewriteGenerator
from far2.data.symbreak import SymBreakGenerator
from far2.data.transfer import (CongruenceGenerator, IterApplyGenerator,
                                RationalTrapGenerator, SymPolyGenerator)
from far2.data.vocab import MAX_SIG

GENERATORS: dict[str, TaskGenerator] = {g.name: g for g in [
    CarryAddGenerator(), MultiplyGenerator(), BaseArithGenerator(),
    ExprEvalGenerator(), LinSysGenerator(), SymBreakGenerator(),
    RewriteGenerator(), DifferentiateGenerator(), ExtraneousGenerator(),
    OptimizeGenerator(),
]}

TRANSFER_GENERATORS: dict[str, TaskGenerator] = {g.name: g for g in [
    SymPolyGenerator(), RationalTrapGenerator(), IterApplyGenerator(),
    CongruenceGenerator(),
]}

ALL_GENERATORS: dict[str, TaskGenerator] = {**GENERATORS, **TRANSFER_GENERATORS}
assert set(ALL_GENERATORS) <= set(FAMILY_ORDER), "FAMILY_ORDER out of sync"

CORE_FAMILIES = ["carry_add", "multiply", "base_arith", "expr_eval",
                 "rewrite", "differentiate"]

# Counterfactual-base protocol (M2): train bases vs held-out test bases.
TRAIN_BASES = (5, 7, 10)
TEST_BASES = (6, 9, 11)


@dataclass
class Mix:
    family: str
    weight: float
    dmin: int
    dmax: int
    choices: tuple | None = None       # explicit difficulty set (e.g. train
                                       # bases {5,7,10}) instead of a range

    def clamp(self) -> "Mix":
        assert self.family in GENERATORS, (
            f"'{self.family}' is a held-out transfer family — it must never "
            "appear in a training mix")
        g = GENERATORS[self.family]
        ch = (tuple(c for c in self.choices
                    if g.min_difficulty <= c <= g.max_difficulty)
              if self.choices else None)
        return Mix(self.family, self.weight,
                   max(self.dmin, g.min_difficulty),
                   min(self.dmax, g.max_difficulty), ch)


class MixSampler:
    """Draws samples family-by-weight, difficulty uniform in the mix's range
    (or from its explicit choices). Deterministic for a given seed regardless
    of batch partitioning. The leading family token is shuffled per episode
    (family_token_mode='random', the default) so it cannot become a controller
    shortcut; 'true' restores the informative token for the leak ablation."""

    def __init__(self, mixes: list[Mix], seed: int = 0,
                 family_token_mode: str = "random"):
        self.set_mixes(mixes)
        self.rng = random.Random(seed)
        self.family_token_mode = family_token_mode

    def set_mixes(self, mixes: list[Mix]) -> None:
        self.mixes = [m.clamp() for m in mixes]
        total = sum(m.weight for m in self.mixes)
        self._cum = []
        acc = 0.0
        for m in self.mixes:
            acc += m.weight / total
            self._cum.append(acc)

    def one(self) -> Sample:
        r = self.rng.random()
        mix = next(m for m, c in zip(self.mixes, self._cum) if r <= c)
        d = (self.rng.choice(mix.choices) if mix.choices
             else self.rng.randint(mix.dmin, mix.dmax))
        s = GENERATORS[mix.family].sample(self.rng, d)
        if self.family_token_mode == "random":
            s = randomize_family_token(s, self.rng)
        return s

    def batch(self, n: int) -> list[Sample]:
        return [self.one() for _ in range(n)]

    def sig_offset(self) -> int:
        """Random Abacus offset for this batch (training only — eval uses 0)."""
        return self.rng.randrange(0, MAX_SIG - 34)


def _show():
    rng = random.Random(0)
    for name, gen in ALL_GENERATORS.items():
        held_out = " · HELD-OUT TRANSFER" if name in TRANSFER_GENERATORS else ""
        mid = (gen.min_difficulty + gen.max_difficulty) // 2
        s = gen.sample(rng, mid)
        print(f"── {name} · difficulty {mid} · {len(s.tokens)} tokens{held_out}")
        print("   " + " ".join(s.tokens))
        print(f"   targets: {' '.join(s.targets)}  "
              f"(solver agrees: {gen.solve(s) == s.targets})")
        print()


if __name__ == "__main__":
    import sys

    if "--show-samples" in sys.argv:
        _show()
    else:
        print("usage: python -m far2.data.registry --show-samples")
