"""Curriculum stages A-F (blueprint Part V, v0.4 mathematical families):
which families train, at what difficulty, with which operators unlocked and
which loss weights active. Training difficulty stays INSIDE the extrapolation
frontier (carry depth <= 6, expression depth <= 6, operand digits <= 3, bases
{5,7,10}, ...) so every phase's OOD evaluation is genuinely out of
distribution. Earlier families keep replay mass so nothing is forgotten.
Phase scripts use single-stage curricula; run_ablations uses FULL_CURRICULUM.

bind_weight anneals the ENTITY + BINDING CURRICULUM (entity value readout and
soft variable->slot supervision): on while operators are being taught, zero
before transfer claims."""

from dataclasses import dataclass, field, replace

from far2.data.registry import CORE_FAMILIES, GENERATORS, TRAIN_BASES, Mix

# training-side difficulty frontier per family (eval goes beyond it)
TRAIN_DMAX = {"carry_add": 6, "multiply": 3, "base_arith": 10, "expr_eval": 6,
              "rewrite": 5, "differentiate": 5, "linsys": 4, "symbreak": 4,
              "extraneous": 6, "optimize": 4}


@dataclass
class Stage:
    name: str
    frac: float                       # fraction of total steps
    mixes: list[Mix]
    unlocked: set[str] = field(default_factory=set)
    depth: str = "fixed"              # fixed | random | heavy
    ipt_alpha: float = 0.0
    oracle_weight: float = 0.0
    bind_weight: float = 0.0          # entity/binding curriculum (anneal to 0)
    slot_cost: float = 0.0
    iter_cost: float = 0.0
    entropy_bonus: float = 0.0
    tau: float = 1.0


def fam(name: str, weight: float, dmax: int | None = None,
        easy: bool = False) -> Mix:
    g = GENERATORS[name]
    cap = dmax if dmax is not None else TRAIN_DMAX.get(name, g.max_difficulty)
    if easy:
        cap = max(g.min_difficulty, cap // 2 + g.min_difficulty // 2)
    if name == "base_arith":
        return Mix(name, weight, g.min_difficulty, cap, choices=TRAIN_BASES)
    return Mix(name, weight, g.min_difficulty, cap)


def core_mixes(weight: float = 1.0, hard: bool = True) -> list[Mix]:
    return [fam(f, weight, easy=not hard) for f in CORE_FAMILIES]


FULL_CURRICULUM: list[Stage] = [
    Stage("A_basic", 0.25, core_mixes(1.0, hard=False), depth="fixed"),
    Stage("B_depth", 0.20, core_mixes(1.0, hard=True), depth="random",
          ipt_alpha=0.5),
    Stage("C_repr", 0.25,
          core_mixes(0.4) + [fam("linsys", 0.6), fam("symbreak", 1.5)],
          unlocked={"expand", "split"}, depth="random", ipt_alpha=0.5,
          oracle_weight=0.5, bind_weight=0.5, slot_cost=0.01,
          entropy_bonus=0.01, tau=0.8),
    Stage("D_reconsider", 0.15,
          core_mixes(0.3) + [fam("linsys", 0.4), fam("symbreak", 0.8),
                             fam("extraneous", 1.5)],
          unlocked={"expand", "split", "halt", "reconsider"}, depth="random",
          ipt_alpha=0.5, oracle_weight=0.5, bind_weight=0.2, slot_cost=0.01,
          iter_cost=0.01, entropy_bonus=0.01, tau=0.7),
    Stage("E_epistemic", 0.10,
          core_mixes(0.3) + [fam("symbreak", 0.6), fam("extraneous", 0.8),
                             fam("linsys", 1.5)],
          unlocked={"expand", "split", "halt", "reconsider", "epistemic"},
          depth="random", ipt_alpha=0.5, slot_cost=0.01, iter_cost=0.01,
          tau=0.6),
    Stage("F_goal", 0.05,
          core_mixes(0.3) + [fam("symbreak", 0.5), fam("extraneous", 0.5),
                             fam("linsys", 0.5), fam("optimize", 1.5)],
          unlocked={"expand", "split", "halt", "reconsider", "epistemic",
                    "goal"},
          depth="random", ipt_alpha=0.5, slot_cost=0.01, iter_cost=0.01,
          tau=0.5),
]


class Curriculum:
    def __init__(self, stages: list[Stage], total_steps: int):
        total_frac = sum(s.frac for s in stages)
        self.bounds: list[tuple[int, Stage]] = []
        acc = 0.0
        for s in stages:
            acc += s.frac / total_frac
            self.bounds.append((int(acc * total_steps), s))

    def at(self, step: int) -> Stage:
        for bound, stage in self.bounds:
            if step < bound:
                return stage
        return self.bounds[-1][1]

    def to_dict(self) -> list[dict]:
        """Serializable, exact stage contract saved beside every checkpoint.

        TrainConfig alone omits stage-level loss weights and operator unlocks;
        without this ledger two behaviorally different runs can appear to have
        the same configuration.
        """
        rows, start = [], 0
        for end, stage in self.bounds:
            rows.append({
                "name": stage.name,
                "start_step": start,
                "end_step": end,
                "fraction": stage.frac,
                "mixes": [{"family": m.family, "weight": m.weight,
                            "dmin": m.dmin, "dmax": m.dmax,
                            "choices": list(m.choices) if m.choices else None}
                           for m in stage.mixes],
                "unlocked": sorted(stage.unlocked),
                "depth": stage.depth,
                "ipt_alpha": stage.ipt_alpha,
                "oracle_weight": stage.oracle_weight,
                "bind_weight": stage.bind_weight,
                "slot_cost": stage.slot_cost,
                "iter_cost": stage.iter_cost,
                "entropy_bonus": stage.entropy_bonus,
                "tau": stage.tau,
            })
            start = end
        return rows


def single_stage(mixes: list[Mix], **overrides) -> list[Stage]:
    return [replace(Stage("only", 1.0, mixes), **overrides)]
