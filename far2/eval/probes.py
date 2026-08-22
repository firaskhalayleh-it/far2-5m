"""Representation probes (v0.4) — the R0/R1/R4 diagnostic ladder.

R0  probe_selfcheck(): the pipeline must recover PLANTED slot encodings at
    >95% and score chance on shuffled labels — otherwise the diagnostic, not
    the model, is broken. Runs before any claim.
R1  pair_value_probe() probes WHERE the information lives: encoder H at the
    variable's anchor, pooled Z, the bound slot (linear AND nonlinear), and
    the wrong slot. The pattern is the diagnosis:
      Case A  H high, pooled high, slots ~ chance  -> known but DISTRIBUTED
      Case B  H high, pooled low                   -> bottleneck destroys it
      Case C  H low                                -> upstream encoder problem
      Case D  linear slot low, nonlinear high      -> present, not linear
R4  the strict gates (bound >= chance+20, bound-wrong >= 15, McNemar) are
    applied by run_probes --strict on the binding substrate.

Task: symbreak (or sym_poly for transfer) — decode each pair variable's
value (0..9) from the slot BOUND to it. Ownership comes from the binding
matrix when the model has one, else from cross-attention mass."""

import random

import torch
import torch.nn.functional as F

from far2.data.base import collate, randomize_family_token
from far2.data.registry import ALL_GENERATORS
from far2.eval.harness import _binding, _binding_from_B
from far2.eval.stats import mcnemar_exact

N_CLASSES = 10                          # variable values are digits 0..9


def fit_linear_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                     l2: float = 1e-2) -> torch.Tensor:
    """Closed-form ridge one-vs-all on frozen features. Returns W [d+1, C]."""
    Xb = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    Y = F.one_hot(y, n_classes).float()
    A = Xb.T @ Xb + l2 * torch.eye(Xb.shape[1])
    return torch.linalg.solve(A, Xb.T @ Y)


def probe_accuracy(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> float:
    Xb = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    return float(((Xb @ W).argmax(-1) == y).float().mean())


def fit_mlp_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int,
                  hidden: int = 64, steps: int = 400, seed: int = 0):
    """Small nonlinear probe (Case D detector). Returns a callable scorer."""
    g = torch.Generator().manual_seed(seed)
    d = X.shape[1]
    w1 = (torch.randn(d, hidden, generator=g) * (2.0 / d) ** 0.5).requires_grad_()
    b1 = torch.zeros(hidden, requires_grad=True)
    w2 = (torch.randn(hidden, n_classes, generator=g) * 0.05).requires_grad_()
    b2 = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([w1, b1, w2, b2], lr=1e-2)
    for _ in range(steps):
        logits = torch.relu(X @ w1 + b1) @ w2 + b2
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    def scorer(Xt, yt):
        with torch.no_grad():
            return float(((torch.relu(Xt @ w1 + b1) @ w2 + b2)
                          .argmax(-1) == yt).float().mean())
    return scorer


def probe_selfcheck(dim: int = 64, n_slots: int = 8, n: int = 400,
                    noise: float = 0.3, seed: int = 0) -> dict:
    """R0 — validate the pipeline on synthetic states with PLANTED encodings:
    slot 2 encodes label A, slot 5 encodes label B, everything else is noise.
    The exact same fit/score path must recover A from slot 2 (> 95%) and must
    NOT beat chance on shuffled labels."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(N_CLASSES, dim, generator=g) * 2.0
    ya = torch.randint(0, N_CLASSES, (n,), generator=g)
    Z = torch.randn(n, n_slots, dim, generator=g) * noise
    Z[:, 2] += means[ya]
    X = Z[:, 2]
    cut = (2 * n) // 3
    W = fit_linear_probe(X[:cut], ya[:cut], N_CLASSES)
    planted = probe_accuracy(W, X[cut:], ya[cut:])
    perm = torch.randperm(n, generator=g)
    Ws = fit_linear_probe(X[:cut], ya[perm][:cut], N_CLASSES)
    shuffled = probe_accuracy(Ws, X[cut:], ya[perm][cut:])
    ok = planted > 0.95 and shuffled < 1.0 / N_CLASSES + 0.15
    return {"planted": planted, "shuffled": shuffled,
            "chance": 1.0 / N_CLASSES, "ok": ok}


@torch.no_grad()
def _collect(model, device, family: str, n: int, difficulty: int, T: int,
             seed: int):
    """Run episodes, return per-row features for every probe location."""
    gen = ALL_GENERATORS[family]
    rng = random.Random(f"probe:{seed}:{family}:{difficulty}")
    rows = {k: [] for k in ("bound", "wrong_slot", "pooled", "H_anchor")}
    labels, episode_of = [], []
    n_episodes = 0
    remaining = n
    while remaining > 0:
        bsz = min(64, remaining)
        remaining -= bsz
        samples = [randomize_family_token(gen.sample(rng, difficulty), rng)
                   for _ in range(bsz)]
        batch = collate(samples, device)
        tel = model(batch, n_iters=T, collect=True).telemetry
        if "Z_final" not in tel:
            raise ValueError("probes need a far2-topology model")
        Zf = tel["Z_final"].cpu()
        Hf = tel["H"].cpu()
        for i, s in enumerate(samples):
            if "pair" not in s.meta or "vars" not in s.meta:
                continue
            ta, tb = s.meta["pair"]
            vals = (s.meta["vars"].get(ta), s.meta["vars"].get(tb))
            if vals[0] is None or vals[1] is None:
                continue
            b = (_binding_from_B(tel["bind_matrix"][i], tel["active_mask"][i],
                                 s, tel.get("n_registers", 0))
                 if "bind_matrix" in tel else
                 _binding(tel["cross_attn"][i], tel["active_mask"][i], s))
            if not b:
                continue
            sa, sb = b["slot_a"], b["slot_b"]
            act = tel["active_mask"][i].cpu()
            pooled = (Zf[i] * act[:, None]).sum(0) / act.sum().clamp_min(1)
            pos = {t: next((k for k, tok in enumerate(s.tokens) if tok == t), 0)
                   for t in (ta, tb)}
            for (feat_a, feat_b), name in (
                    ((Zf[i, sa], Zf[i, sb]), "bound"),
                    ((Zf[i, sb], Zf[i, sa]), "wrong_slot"),
                    ((pooled, pooled), "pooled"),
                    ((Hf[i, pos[ta]], Hf[i, pos[tb]]), "H_anchor")):
                rows[name] += [feat_a, feat_b]
            labels += [vals[0], vals[1]]
            episode_of += [n_episodes, n_episodes]
            n_episodes += 1
    return rows, labels, episode_of, n_episodes


def pair_value_probe(model, device, n: int = 512, difficulty: int = 3,
                     T: int = 8, seed: int = 7,
                     family: str = "symbreak") -> dict:
    """R1 location probes + the separation quantities the gates consume."""
    rows, labels, episode_of, n_episodes = _collect(
        model, device, family, n, difficulty, T, seed)
    if n_episodes < 20:
        return {"n_episodes": n_episodes, "error": "too few probe episodes"}

    split_rng = random.Random(seed)
    test_eps = set(split_rng.sample(range(n_episodes), n_episodes // 3))
    y = torch.tensor(labels)
    is_test = torch.tensor([e in test_eps for e in episode_of])
    out = {"n_episodes": n_episodes, "chance": 1.0 / N_CLASSES}
    outcomes = {}
    for name, feats in rows.items():
        X = torch.stack(feats).float()
        W = fit_linear_probe(X[~is_test], y[~is_test], N_CLASSES)
        Xt = torch.cat([X[is_test], torch.ones(int(is_test.sum()), 1)], dim=1)
        correct = (Xt @ W).argmax(-1) == y[is_test]
        outcomes[name] = correct
        out[name] = float(correct.float().mean())
        out[f"{name}_k"] = int(correct.sum())
        out[f"{name}_n"] = int(correct.numel())
    # nonlinear probe on the bound slot (Case D detector)
    Xb = torch.stack(rows["bound"]).float()
    scorer = fit_mlp_probe(Xb[~is_test], y[~is_test], N_CLASSES, seed=seed)
    out["bound_mlp"] = scorer(Xb[is_test], y[is_test])

    out["separation_index"] = out["bound"] - out["wrong_slot"]
    b = int((outcomes["bound"] & ~outcomes["wrong_slot"]).sum())
    c = int((~outcomes["bound"] & outcomes["wrong_slot"]).sum())
    out["mcnemar_b"], out["mcnemar_c"] = b, c
    out["mcnemar_p"] = mcnemar_exact(b, c)
    out["diagnosis"] = _diagnose(out)
    return out


def _diagnose(r: dict) -> str:
    ch = r["chance"]
    if r["H_anchor"] < ch + 0.15:
        return ("Case C: the anchor value is not decodable even from the "
                "ENCODER — the problem is upstream of the workspace")
    if r["pooled"] < ch + 0.15:
        return ("Case B: encoder knows it, pooled workspace does not — the "
                "slot bottleneck destroys the information")
    if r["bound"] < ch + 0.10 and r["bound_mlp"] >= ch + 0.20:
        return ("Case D: the concept lives in the bound slot but NOT in a "
                "linearly separable geometry")
    if r["bound"] < ch + 0.10:
        return ("Case A: the workspace knows the value but representation is "
                "DISTRIBUTED across slots — slot != concept; the substrate, "
                "not the operators, is the research problem")
    return "localized: the bound slot linearly encodes its variable's value"
