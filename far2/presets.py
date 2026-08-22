"""Named model configs. The three 5M presets are budget-matched (each asserted
within 4.2–5.8M by tests); `*_tiny` variants exist for tests and the smoke run.
Ablation presets differ from far2_5m by exactly one mechanism each."""

from dataclasses import replace

from far2.config import ModelConfig


def transformer_5m() -> ModelConfig:
    return ModelConfig(kind="transformer", d_model=256, n_heads=4, mlp_mult=4,
                       n_layers=6)


def recurrent_5m() -> ModelConfig:
    return ModelConfig(kind="recurrent", d_model=512, n_heads=4, mlp_mult=6)


def far2_5m() -> ModelConfig:
    return ModelConfig(kind="far2", d_model=320, n_heads=4, mlp_mult=4,
                       max_slots=16, init_slots=4)


def slot_recurrent_5m() -> ModelConfig:
    """Baseline C: identical topology to FAR² (encode once → slot workspace →
    cross-attend H → decode) with every FAR² operator off and 16 fixed slots.
    D − C is the FAR² contribution; without C, a win could be credited to the
    recurrent latent bottleneck alone."""
    return replace(far2_5m(), init_slots=16, max_slots=16, n_iters_train=8)


def far2_full() -> ModelConfig:
    return replace(far2_5m(), adaptive_slots=True, halting="learned",
                   reconsider=True, epistemic=True, protected_goal=True)


# ---- ablation grid (each isolates one mechanism vs far2_full) -------------
def far2_no_adaptive() -> ModelConfig:
    """Fixed-max-slots: all 16 always on, controller absent."""
    return replace(far2_full(), adaptive_slots=False, init_slots=16)


def far2_fixed4() -> ModelConfig:
    """Fixed-min-slots at far2_full's slot budget: 4 slots forever, growth off.
    The capacity floor of the necessity-vs-allocation quartet."""
    return replace(far2_full(), adaptive_slots=False, init_slots=4)


def far2_min_slots() -> ModelConfig:
    return replace(far2_full(), adaptive_slots=False, init_slots=4, max_slots=4)


def far2_no_reconsider() -> ModelConfig:
    return replace(far2_full(), reconsider=False)


def far2_no_epistemic() -> ModelConfig:
    return replace(far2_full(), epistemic=False)


def far2_no_goal() -> ModelConfig:
    return replace(far2_full(), protected_goal=False)


def far2_fixed_recur() -> ModelConfig:
    return replace(far2_full(), halting="fixed")


# ---- v0.3 causal controls (isolate the alternative explanations) ----------
def far2_random_expand() -> ModelConfig:
    """EXPAND/SPLIT at rate-matched RANDOM times with a RANDOM split source —
    param-identical to far2_full. If this matches the learned controller, the
    win was extra slots/noise, not decisions. Scripts overwrite the rates with
    the learned arm's measured firing rates."""
    return replace(far2_full(), repr_policy="random")


def far2_oracle_expand() -> ModelConfig:
    """SPLIT forced by the task condition (perfect decisions) — the upper
    bound on what representation adaptation COULD deliver. Consults ground
    truth at run time: report as a bound, never as a win."""
    return replace(far2_full(), repr_policy="oracle")


def far2_no_split() -> ModelConfig:
    """EXPAND allowed, SPLIT never unlocks (fine-grained action ablation)."""
    return replace(far2_full(), disable_actions=("split",))


def far2_random_reset() -> ModelConfig:
    """Partial reset at rate-matched RANDOM times: isolates intelligent
    reconsideration from generic state perturbation. Fixed budget by design."""
    return replace(far2_full(), halting="fixed", reconsider_policy="random")


def far2_goal_scrambled() -> ModelConfig:
    """Goal channel intact (same params, same compute) but fed another
    episode's goal: isolates 'protected goal' from 'extra clean channel'."""
    return replace(far2_full(), goal_scrambled=True)


# ---- v0.4 binding substrate (R3+: Research Problem #1) --------------------
def far2_bind() -> ModelConfig:
    """far2_full on the competitive-binding workspace: evidence tokens choose
    their owner slot (Slot-Attention bias), 2 register slots absorb
    background, the variable->slot binding matrix drives collision detection
    and directed SPLIT. The substrate that makes 'slot = concept' testable."""
    return replace(far2_full(), cross_mode="competitive", n_registers=2,
                   binding=True, collision_signal="bind")


def far2_bind_random_expand() -> ModelConfig:
    """Rate-matched random-policy control on the SAME binding substrate —
    param-identical to far2_bind."""
    return replace(far2_bind(), repr_policy="random")


def far2_bind_oracle() -> ModelConfig:
    """Condition-forced upper bound on the binding substrate (R2/R5)."""
    return replace(far2_bind(), repr_policy="oracle")


def slot_recurrent_bind() -> ModelConfig:
    """Baseline C on the competitive workspace — PARAM-IDENTICAL control for
    far2_bind: the FAR² modules exist (and cost their parameters) but every
    operator is disabled, so behaviorally it is a fixed 14+2-slot competitive
    workspace. Built from far2_bind, not slot_recurrent_5m, so 'D beats C' can
    never be a 435k-parameter artifact (review blocker: 4,980,563 vs
    4,545,992 under a param-matched claim)."""
    return replace(far2_bind(), init_slots=14, n_iters_train=8,
                   disable_actions=("expand", "split", "halt", "reconsider",
                                    "epistemic", "goal"))


# ---- tiny variants for tests / smoke --------------------------------------
def _tiny(cfg: ModelConfig) -> ModelConfig:
    over = dict(d_model=64, n_heads=2, controller_hidden=32, goal_dim=16)
    if cfg.kind == "transformer":
        over["n_layers"] = 2
    if cfg.kind == "far2":
        over["max_slots"] = 8
    return replace(cfg, **over)


PRESETS = {
    "transformer_5m": transformer_5m, "recurrent_5m": recurrent_5m,
    "slot_recurrent_5m": slot_recurrent_5m,
    "far2_5m": far2_5m, "far2_full": far2_full,
    "far2_no_adaptive": far2_no_adaptive, "far2_fixed4": far2_fixed4,
    "far2_min_slots": far2_min_slots,
    "far2_no_reconsider": far2_no_reconsider, "far2_no_epistemic": far2_no_epistemic,
    "far2_no_goal": far2_no_goal, "far2_fixed_recur": far2_fixed_recur,
    "far2_random_expand": far2_random_expand, "far2_oracle_expand": far2_oracle_expand,
    "far2_no_split": far2_no_split, "far2_random_reset": far2_random_reset,
    "far2_goal_scrambled": far2_goal_scrambled,
    "far2_bind": far2_bind, "far2_bind_random_expand": far2_bind_random_expand,
    "far2_bind_oracle": far2_bind_oracle,
    "slot_recurrent_bind": slot_recurrent_bind,
    "transformer_tiny": lambda: _tiny(transformer_5m()),
    "recurrent_tiny": lambda: _tiny(recurrent_5m()),
    "slot_recurrent_tiny": lambda: _tiny(slot_recurrent_5m()),
    "far2_tiny": lambda: _tiny(far2_5m()),
    "far2_tiny_full": lambda: _tiny(far2_full()),
    "far2_tiny_bind": lambda: _tiny(far2_bind()),
}

# Controls that must cost exactly what their reference model costs (their
# modules exist, some are simply never consulted). Asserted by tests.
PARAM_IDENTICAL_TO_FULL = ("far2_random_expand", "far2_oracle_expand",
                           "far2_no_split", "far2_random_reset",
                           "far2_goal_scrambled")
PARAM_IDENTICAL_TO_BIND = ("far2_bind_random_expand", "far2_bind_oracle",
                           "slot_recurrent_bind")


def get_preset(name: str) -> ModelConfig:
    return PRESETS[name]()


def param_table(preset_names=None) -> str:
    """Per-module parameter ledger straight from the code — the authority the
    blueprint tables are generated from. CLI: python -m far2.presets --table"""
    from far2.models import build_model
    lines = []
    for name in preset_names or ["transformer_5m", "recurrent_5m",
                                 "slot_recurrent_5m", "far2_5m", "far2_full"]:
        model = build_model(get_preset(name))
        lines.append(f"\n{name}  ·  total {model.param_count():,}")
        for child, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n:
                lines.append(f"  {child:<12} {n:>12,}")
        direct = sum(p.numel() for p in model.parameters(recurse=False))
        if direct:
            lines.append(f"  {'(direct)':<12} {direct:>12,}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if "--table" in sys.argv:
        print(param_table())
    else:
        print("usage: python -m far2.presets --table")
