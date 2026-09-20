"""Sample container, generator interface, and the collate that turns samples
into the tensor contract every model consumes (blueprint Part II).

v0.4: numbers are digit runs, so collate also emits `digit_sig` (the shared
Abacus-style significance index every model embeds) and syntax-derived
`var_pos`/`var_occurrence`/`var_rel` tensors. Ground-truth `var_values` are a
training-only binding-curriculum label; runtime structure is derived solely
from the rendered token sequence."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from far2.data.vocab import MAX_SIG, N_VARS, VOCAB, digit_significance

# Canonical family order for telemetry ids (registry asserts it stays in sync).
# The FAMILY TOKEN in the sequence is shuffled per episode and carries no task
# information — these names, not the token, identify the family in records.
FAMILY_ORDER = ["carry_add", "multiply", "base_arith", "expr_eval", "linsys",
                "symbreak", "rewrite", "differentiate", "extraneous",
                "optimize",
                # held-out transfer families (never trained on):
                "sym_poly", "rational_trap", "iter_apply", "congruence"]


@dataclass
class Sample:
    family: str
    difficulty: int
    tokens: list[str]                  # includes the ANS slots
    targets: list[str]                 # one per ANS occurrence, in order
    epistemic: int = -1                # config.DERIVED/... or -1
    requires_distinction: int = -1     # 1 = solving requires separating a
                                       #   symmetric/entangled pair; 0 = matched
                                       #   control. A CURRICULUM/EVAL condition
                                       #   on the task — NOT an oracle claim
                                       #   about the model's internal state
                                       #   (that needs binding diagnostics).
    reconsider_label: int = -1         # 1 = tempting candidate invalid under the
                                       #   original constraints (task condition)
    meta: dict = field(default_factory=dict)
    # meta conventions (harness copies these into episode records):
    #   D: reasoning depth   R: representation demand   (L = len(tokens))
    #   vars: {"X3": value, ...} ground-truth variable values (digits 0..9)
    #   pair: ("X0","X1") the pair binding diagnostics track
    #   attractor: "D4" the tempting-candidate token (reconsider families)

    def __post_init__(self):
        n_ans = sum(1 for t in self.tokens if t == "ANS")
        assert n_ans == len(self.targets), (
            f"{self.family}: {n_ans} ANS slots vs {len(self.targets)} targets")


class TaskGenerator(ABC):
    """A task family. sample() builds a Sample; solve() recomputes the targets
    from the rendered context alone via an EXACT solver — the pytest ground-
    truth audit asserts sample.targets == solve(sample) so targets can never
    silently drift. residual() (optional) scores a candidate decode with the
    exact checker, for per-iteration progress telemetry only — models never
    see it."""

    name: str
    family_token: str
    min_difficulty: int = 1
    max_difficulty: int = 8

    @abstractmethod
    def sample(self, rng: random.Random, difficulty: int, **kw) -> Sample: ...

    @abstractmethod
    def solve(self, sample: Sample) -> list[str]: ...

    def score(self, sample: Sample, pred_tokens: list[str]) -> dict:
        """Per-episode extras beyond exact match; families override to add
        e.g. constraint violations or assertion flags."""
        return {"correct": pred_tokens == sample.targets}

    def residual(self, sample: Sample, pred_tokens: list[str]) -> float | None:
        return None


def randomize_family_token(sample: Sample, rng: random.Random) -> Sample:
    """Shuffle the leading family token so it cannot become a controller
    shortcut (e.g. 'F5 -> SPLIT'). Applied by the sampler AND the harness so
    train and eval distributions match; telemetry uses Sample.family."""
    assert sample.tokens[0].startswith("F"), "family token must lead the sample"
    sample.tokens[0] = f"F{rng.randrange(10)}"
    return sample


def collate(samples: list[Sample], device: torch.device | None = None,
            sig_offset: int = 0) -> dict:
    """sig_offset shifts every nonzero significance index (Abacus trick):
    training draws a random offset per batch so ALL significance embeddings
    receive gradient and unseen lengths extrapolate; eval uses 0."""
    v = VOCAB
    pad, ans = v.t("PAD"), v.t("ANS")
    B = len(samples)
    L = max(len(s.tokens) for s in samples)
    A = max(len(s.targets) for s in samples)

    input_ids = torch.full((B, L), pad, dtype=torch.long)
    digit_sig = torch.zeros((B, L), dtype=torch.long)
    pad_mask = torch.zeros((B, L), dtype=torch.bool)
    goal_mask = torch.zeros((B, L), dtype=torch.bool)
    ans_pos = torch.full((B, A), -1, dtype=torch.long)
    answer_query_ids = torch.full((B, A), ans, dtype=torch.long)
    answer_query_vars = torch.full((B, A), -1, dtype=torch.long)
    target_ids = torch.full((B, A), -100, dtype=torch.long)
    family_id = torch.zeros(B, dtype=torch.long)
    difficulty = torch.zeros(B, dtype=torch.long)
    epistemic = torch.full((B,), -1, dtype=torch.long)
    requires_distinction = torch.full((B,), -1, dtype=torch.long)
    reconsider_label = torch.full((B,), -1, dtype=torch.long)
    var_pos = torch.full((B, N_VARS), -1, dtype=torch.long)
    var_occurrence = torch.zeros((B, N_VARS, L), dtype=torch.bool)
    var_rel = torch.zeros((B, N_VARS, N_VARS), dtype=torch.bool)
    var_values = torch.full((B, N_VARS), -100, dtype=torch.long)
    pair_idx = torch.full((B, 2), -1, dtype=torch.long)
    oracle_split_pairs = torch.full((B, N_VARS // 2, 2), -1,
                                    dtype=torch.long)
    oracle_split_count = torch.zeros(B, dtype=torch.long)

    for b, s in enumerate(samples):
        ids = v.encode(s.tokens)
        input_ids[b, : len(ids)] = torch.tensor(ids)
        pad_mask[b, : len(ids)] = True
        sig = digit_significance(s.tokens)
        digit_sig[b, : len(ids)] = torch.tensor(
            [min(x + sig_offset, MAX_SIG) if x else 0 for x in sig])
        positions = [i for i, t in enumerate(ids) if t == ans]
        ans_pos[b, : len(positions)] = torch.tensor(positions)
        target_ids[b, : len(s.targets)] = torch.tensor(v.encode(s.targets))
        # FAR2 output queries are syntax-only: answer rank plus, when the
        # grammar provides one query token per answer, that token's identity.
        # They never reuse contextual encoder states at ANS positions, so the
        # final answer must read the recurrent workspace.
        if "Q" in s.tokens:
            q0 = s.tokens.index("Q") + 1
            a0 = s.tokens.index("ANS")
            query_tokens = s.tokens[q0:a0]
            if len(query_tokens) == len(s.targets):
                answer_query_ids[b, : len(query_tokens)] = torch.tensor(
                    v.encode(query_tokens))
                for ai, tok in enumerate(query_tokens):
                    if tok.startswith("X") and tok[1:].isdigit():
                        answer_query_vars[b, ai] = int(tok[1:])
        family_id[b] = FAMILY_ORDER.index(s.family)   # token is shuffled; name isn't
        difficulty[b] = s.difficulty
        epistemic[b] = s.epistemic
        requires_distinction[b] = s.requires_distinction
        reconsider_label[b] = s.reconsider_label
        # Runtime-legitimate variable structure comes from rendered tokens,
        # never from meta.vars or the declared evaluation pair.
        for i, tok in enumerate(s.tokens):
            if tok.startswith("X") and tok[1:].isdigit():
                vi = int(tok[1:])
                if 0 <= vi < N_VARS:
                    var_occurrence[b, vi, i] = True
                    if var_pos[b, vi] < 0:
                        var_pos[b, vi] = i

        # Relation edges are all variable pairs co-occurring in an input
        # clause (SEP-delimited, before Q). This is observable syntax, not the
        # probe's declared pair or requires_distinction label.
        qpos = s.tokens.index("Q") if "Q" in s.tokens else len(s.tokens)
        clause: set[int] = set()
        for tok in s.tokens[:qpos] + ["SEP"]:
            if tok == "SEP":
                for vi in clause:
                    for vj in clause:
                        if vi != vj:
                            var_rel[b, vi, vj] = True
                clause = set()
            elif tok.startswith("X") and tok[1:].isdigit():
                vi = int(tok[1:])
                if 0 <= vi < N_VARS:
                    clause.add(vi)

        # Ground-truth values supervise the annealed entity/binding curriculum
        # only. They are not consulted by the model's runtime decisions.
        for name, val in s.meta.get("vars", {}).items():
            vi = int(name[1:])
            if 0 <= val <= 9:
                var_values[b, vi] = v.t(f"D{val}")
        if "pair" in s.meta:
            pair_idx[b, 0] = int(s.meta["pair"][0][1:])
            pair_idx[b, 1] = int(s.meta["pair"][1][1:])
        split_pairs = s.meta.get("split_pairs", ())
        oracle_split_count[b] = min(len(split_pairs), N_VARS // 2)
        for pi, pair in enumerate(split_pairs[: N_VARS // 2]):
            oracle_split_pairs[b, pi, 0] = int(pair[0][1:])
            oracle_split_pairs[b, pi, 1] = int(pair[1][1:])
        # goal span: GOAL ... up to (excluding) Q — pooled by the protected-goal head
        if "GOAL" in s.tokens:
            g0 = s.tokens.index("GOAL")
            g1 = s.tokens.index("Q", g0)
            goal_mask[b, g0:g1] = True

    batch = {
        "input_ids": input_ids, "digit_sig": digit_sig,
        "pad_mask": pad_mask, "goal_mask": goal_mask,
        "ans_pos": ans_pos, "answer_query_ids": answer_query_ids,
        "answer_query_vars": answer_query_vars, "target_ids": target_ids,
        "family_id": family_id, "difficulty": difficulty,
        "epistemic_label": epistemic, "requires_distinction": requires_distinction,
        "reconsider_label": reconsider_label,
        "var_pos": var_pos, "var_occurrence": var_occurrence,
        "var_rel": var_rel, "var_values": var_values, "pair_idx": pair_idx,
        "oracle_split_pairs": oracle_split_pairs,
        "oracle_split_count": oracle_split_count,
        "samples": samples,                      # kept for scoring; not a tensor
    }
    if device is not None:
        batch = {k: (t.to(device) if isinstance(t, torch.Tensor) else t)
                 for k, t in batch.items()}
    return batch
