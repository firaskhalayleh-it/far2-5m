"""Phase 2 — mathematical extrapolation. Trained with random T ∈ {2..8} on
in-frontier difficulty (carry depth ≤6, expression depth ≤6, ≤3-digit
multiplication), evaluated BEYOND the frontier on the extrapolation matrix:

    carry depth   {2, 6, 12, 24}   (trained ≤ 6)
    digit length  {8, 32}          matched counterexample: same L, D=0 vs D=24
    expr depth    {4, 8, 12, 16}   (trained ≤ 6)
    multiply      {2, 3, 4, 6}     (trained ≤ 3)
    deriv depth   {4, 8, 12}       (trained ≤ 5)

read out at T ∈ {2,4,8,16,32}. The claim under test: inference recurrence
extrapolates mathematical computation beyond trained depth — and LENGTH alone
must not be what costs accuracy. The 8>16>32 failure signal triggers the
pre-registered contingency ladder, not a panic. See docs/ROADMAP.md."""

from dataclasses import replace

from _common import apply_mini, finish, make_tc, parser, run_eval, train

from far2.eval import EvalSpec
from far2.eval.metrics import Gate, accuracy, residual_monotonicity
from far2.presets import get_preset
from far2.train import core_mixes, single_stage

# family -> (eval difficulties = depth axis, OOD threshold for the hard bucket)
DEPTH_FAMILIES = {"carry_add": ([2, 6, 12, 24], 8),
                  "expr_eval": ([4, 8, 12, 16], 8),
                  "multiply": ([2, 3, 4, 6], 4),
                  "differentiate": ([4, 8, 12], 8)}
EVAL_T = [2, 4, 8, 16, 32]


def main():
    p = parser("Phase 2: variable depth + mathematical extrapolation",
               steps=4000)
    p.add_argument("--depth", choices=["random", "heavy"], default="random",
                   help="'heavy' is contingency ladder rung 1 (log-normal tail)")
    args = p.parse_args()
    apply_mini(args)

    mc = replace(get_preset("far2_5m"), init_slots=8, max_slots=8, n_iters_train=8)
    tc = make_tc(args, f"phase2_{args.depth}_seed{args.seed}", ipt_alpha=0.5)
    trainer = train(mc, tc, single_stage(core_mixes(1.0, hard=True),
                                         depth=args.depth, ipt_alpha=0.5))

    specs = [EvalSpec(f, ds, EVAL_T, args.eval_samples)
             for f, (ds, _) in DEPTH_FAMILIES.items()]
    recs = run_eval(trainer, specs, "depth_grid")
    # matched counterexample: 32 digits with NO carries vs a 24-deep chain —
    # same surface length, different reasoning depth
    matched = run_eval(trainer,
                       [EvalSpec("carry_add", [0], [8], args.eval_samples,
                                 {"n_digits": 32}),
                        EvalSpec("carry_add", [24], [8], args.eval_samples,
                                 {"n_digits": 32})], "length_match")

    hard = {f: h for f, (_, h) in DEPTH_FAMILIES.items()}
    def acc_at(T, bucket):
        sel = [r for r in recs if r["n_iters"] == T
               and ((r["difficulty"] >= hard[r["family"]]) == (bucket == "hard"))]
        return sum(r["correct"] for r in sel) / len(sel) if sel else float("nan")

    print("\naccuracy = f(iterations)   (hard = beyond the training frontier):")
    print(f"  {'T':>4} {'in-dist':>8} {'OOD':>8}")
    for T in EVAL_T:
        print(f"  {T:>4} {acc_at(T, 'easy'):>8.1%} {acc_at(T, 'hard'):>8.1%}")

    m_easy = accuracy(matched, difficulty=0)
    m_hard = accuracy(matched, difficulty=24)
    print(f"\n  matched counterexample @32 digits, T=8: "
          f"no-carry {m_easy:.1%} vs 24-chain {m_hard:.1%} "
          f"(length is identical — only reasoning depth differs)")
    mono = residual_monotonicity(recs)
    if mono.get("n"):
        print(f"  exact-progress: {mono['frac_monotone']:.1%} episodes with "
              f"non-increasing residual (n={mono['n']})")

    gate = Gate("Phase 2 · mathematical depth extrapolation")
    e2 = acc_at(2, "easy")
    gate.check("in-distribution ~ solved by T=2..4", e2 >= 0.60,
               f"in-dist@2 {e2:.1%}")
    h2, h4, h8, h16, h32 = (acc_at(T, "hard") for T in EVAL_T)
    gate.check("OOD: 2 < 4 (>= +3pts)", h4 >= h2 + 0.03, f"{h2:.1%} -> {h4:.1%}")
    gate.check("OOD: 4 < 8 (>= +3pts)", h8 >= h4 + 0.03, f"{h4:.1%} -> {h8:.1%}")
    gate.check("OOD: 16 holds beyond trained depth", h16 >= h8 - 0.02,
               f"8:{h8:.1%} -> 16:{h16:.1%}")
    gate.check("length alone is not the cost (no-carry@32digits stays high)",
               m_easy >= m_hard + 0.05 or m_easy >= 0.9,
               f"L-matched: D0 {m_easy:.1%} vs D24 {m_hard:.1%}")
    if h8 - h32 > 0.10:
        print("\n  FAILURE SIGNAL: accuracy degrades sharply past trained depth "
              f"(8:{h8:.1%} -> 32:{h32:.1%}).\n  This is research problem #1. "
              "Pre-registered remedy ladder (docs/DESIGN.md):\n"
              "    1. --depth heavy (log-normal tail T sampling)\n"
              "    2. TrainConfig.fp_penalty=0.1 (terminal fixed-point pull)\n"
              "    3. JSRR (jacobian reg)  4. ModelConfig.gate_style='convex'")
    finish(gate.report())


if __name__ == "__main__":
    main()
