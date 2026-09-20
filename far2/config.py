"""All knobs for models and training. Values here are the source of truth; the
blueprint document is the map. Defaults follow the research calibration recorded
in docs/DESIGN.md."""

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    kind: str = "far2"                # transformer | recurrent | far2
    vocab_size: int = 72              # far2.data.vocab (v0.4 mathematical table)
    d_model: int = 320
    n_heads: int = 4
    mlp_mult: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0

    # numbers (v0.4): shared Abacus-style digit-significance embedding — ALL
    # models get the identical numeric representation, so FAR² can never win
    # by beating a deliberately weak arithmetic baseline
    digit_sig: bool = True

    # transformer baseline
    n_layers: int = 6

    # recurrence (recurrent baseline + far2)
    n_iters_train: int = 4
    n_iters_max: int = 32
    iter_embedding: bool = False      # timestep embeddings break depth extrapolation
    recall: bool = True               # re-inject input every iteration (recurrent baseline)

    # far2: structure
    n_encoder_layers: int = 1
    max_slots: int = 16
    init_slots: int = 4
    slot_jitter: float = 0.02         # per-forward noise on seeds at activation
    workspace_init: str = "learned"   # learned | input | input_shuffled
                                      # input reuses the core cross-attention to
                                      # warm active slots from H before cycle 1;
                                      # shuffled is the causal content control
    goal_dim: int = 64

    # far2: mechanisms (the ablation flags)
    adaptive_slots: bool = False      # O3/O4 EXPAND+SPLIT controller
    halting: str = "fixed"            # fixed | learned | converge
    reconsider: bool = False          # O6 (requires halting == "learned")
    epistemic: bool = False           # O7
    protected_goal: bool = False      # O8

    # far2: representation substrate (v0.4 R3 — Research Problem #1)
    cross_mode: str = "standard"      # standard: each slot independently
                                      #   softmaxes over H (slots may all grab
                                      #   the same evidence)
                                      # competitive: softmax over SLOTS per
                                      #   evidence token — each token chooses
                                      #   its owner (Slot-Attention bias, the
                                      #   pressure that makes slots specialize)
    n_registers: int = 0              # always-active background slots that
                                      #   absorb non-conceptual content; never
                                      #   SPLIT sources, excluded from
                                      #   collision pair search
    binding: bool = False             # variable->slot binding matrix
                                      #   B[e,s] = softmax_s(E_e^T W_B z_s):
                                      #   E pools all entity occurrences and
                                      #   reads their relational context from H
                                      #   enables collision_signal="bind",
                                      #   binding-directed SPLIT, the binding
                                      #   curriculum loss, and ownership-based
                                      #   diagnostics
    oracle_action: str = "split"      # repr_policy=="oracle": forced action

    # far2: stability (see docs/DESIGN.md)
    gate_style: str = "additive"      # additive: Z+G*dZ | convex: (1-G)*Z+G*dZ
    gate_bias_init: float = -2.0
    state_norm: str = "post_rms"      # post_rms | none
    core_act: str = "relu"            # recurrent-core MLP activation (Lipschitz-friendly)

    # far2: operator details
    split_impl: str = "legacy_clone"  # legacy_clone | semantic
                                      # semantic requires binding and creates
                                      # two exclusive entity-owned children
    split_beta: float = 0.2           # clone noise scale x d_nearest
    reconsider_alpha: float = 0.3     # Z <- a*Z + (1-a)*seeds on RECONSIDER
    controller_hidden: int = 128
    converge_tol: float = 0.1         # halting=="converge": stop when ||dZ||inf/||Z||inf < tol
    collision_signal: str = "C"       # controller signal: C/sim/jsd/attn_iou/
                                      #   bind/entity_relation/binding_overlap
                                      #   (P4-selected)
                                      # C | sim | jsd | attn_iou | attn_iou_x_jsd
                                      # (Phase 4 tournament selects; P3b consumes)
    slot_epistemic: bool = False      # per-slot type tags are NOT supervised at 5M;
                                      # answer-level epistemics is the mechanism

    # far2: scientific controls (v0.3) — each isolates one alternative
    # explanation for a FAR² win; see docs/DESIGN.md "v0.3 causal controls"
    repr_policy: str = "learned"      # learned | random (rate-matched, task-blind)
                                      #         | oracle (task-condition upper bound)
    random_expand_p: float = 0.06     # repr_policy=="random": per-iteration P(EXPAND)
    random_split_p: float = 0.04      # repr_policy=="random": per-iteration P(SPLIT)
    reconsider_policy: str = "learned"  # learned | random (reset-at-random control)
    random_reconsider_p: float = 0.10   # reconsider_policy=="random": per-iter P(reset)
    goal_scrambled: bool = False      # goal channel intact (same params/compute) but
                                      # fed a same-distribution WRONG goal — isolates
                                      # "protected goal" from "extra clean channel"
    disable_actions: tuple = ()       # actions the curriculum can never unlock
                                      # (fine-grained ablations: ("split",) etc.)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 128
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    ema: float = 0.999                # 0 disables; eval prefers EMA weights
    seed: int = 0
    device: str = "auto"
    log_every: int = 50
    eval_every: int = 0               # 0 = only at the end
    eval_samples: int = 200
    out_dir: str = "runs/run"
    checkpoint_steps: tuple = ()      # optional 1-based learning-curve points

    # depth sampling
    depth: str = "fixed"              # fixed | random | heavy  (heavy = contingency ladder rung 1)
    depth_choices: tuple = (2, 3, 4, 5, 6, 8)

    # loss weights (0 disables a term; see blueprint Part VI ledger)
    ipt_alpha: float = 0.0            # intermediate-loss blend weight
    cycle_supervision: str = "all"    # all | random (legacy IPT) | final
                                      # all supervises every pre-final cycle;
                                      # final disables intermediate losses
    state_norm_penalty: float = 0.01
    fp_penalty: float = 0.0           # contingency: ||Z_T - Z_{T-1}||^2
    epistemic_weight: float = 0.5
    oracle_weight: float = 0.0        # controller supervision (curriculum anneals)
    halt_q_weight: float = 0.5
    slot_cost: float = 0.0
    iter_cost: float = 0.0
    entropy_bonus: float = 0.0
    gumbel_tau: float = 1.0           # curriculum anneals toward 0.5
    epistemic_shuffled: bool = False  # v0.3 control: epistemic aux labels permuted
                                      # within batch — same head, same loss pressure,
                                      # zero epistemic content (isolates the mechanism
                                      # from generic aux-loss regularization)
    bind_weight: float = 0.0          # binding-curriculum CE (stage overrides;
                                      # annealed to 0 before any transfer claim)

    def to_dict(self) -> dict:
        return asdict(self)


# Epistemic label codes shared by generators, model, and metrics (v0.4:
# mathematical semantics, not fuzzy confidence).
#   OBSERVED       the answer is read directly off a given statement
#   DERIVED        the answer follows by exact computation/inference
#   INCONSISTENT   the constraints contradict — correct output: NO_SOLUTION
#   UNDERDET       the constraints underdetermine the query — correct
#                  output: UNDERDETERMINED, never a guess
OBSERVED, DERIVED, INCONSISTENT, UNDERDET = 0, 1, 2, 3
EPISTEMIC_NAMES = {OBSERVED: "OBSERVED", DERIVED: "DERIVED",
                   INCONSISTENT: "INCONSISTENT", UNDERDET: "UNDERDETERMINED",
                   -1: "NA"}
