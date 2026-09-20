"""Episode harness (blueprint Part VII): runs a model over a grid of
(family x difficulty x iteration-count) cells and writes one JSONL record per
episode in the telemetry schema the gates and dashboard aggregate.

v0.4 additions: records carry the surface coordinates D (reasoning depth),
R (representation demand), L (surface length) from generator meta — the axes
of the Mathematical Generalization Surface; when the generator ships an exact
residual() checker and per-iteration probe answers exist, records carry
`residual_traj` (does latent cognition become mathematically more correct as
iterations increase? — telemetry only, the model never sees it); binding
diagnostics prefer the ownership matrix B over attention mass when the model
has one. EvalSpec.gen_kwargs passes independent surface knobs (e.g. n_digits)
for matched counterexamples.

Held-out transfer families are reachable here (and only here). The family
token is shuffled per episode by default, matching the training distribution,
so no evaluation can be solved by reading the task label."""

import random
from dataclasses import dataclass, field

import torch

from far2.data.base import collate, randomize_family_token
from far2.data.registry import ALL_GENERATORS
from far2.data.vocab import VOCAB
from far2.utils import JsonlLogger


@dataclass
class EvalSpec:
    family: str
    difficulties: list[int]
    n_iters: list = field(default_factory=lambda: [None])   # None = model default
    n_samples: int = 100
    gen_kwargs: dict = field(default_factory=dict)  # e.g. {"n_digits": 32}


@torch.no_grad()
def evaluate(model, specs: list[EvalSpec], device, seed: int = 1234,
             batch_size: int = 64, out_path: str | None = None,
             family_token_mode: str = "random",
             intervene: dict | None = None) -> list[dict]:
    model.eval()
    logger = JsonlLogger(out_path) if out_path else None
    records: list[dict] = []

    for spec in specs:
        gen = ALL_GENERATORS[spec.family]
        for d in spec.difficulties:
            d = max(gen.min_difficulty, min(d, gen.max_difficulty))
            for T in spec.n_iters:
                # PAIRED across T: the seed excludes T, so every iteration
                # count sees the IDENTICAL episode sequence — "more compute
                # helped" must mean it helped the same problems
                rng = random.Random(f"{seed}:{spec.family}:{d}")
                remaining = spec.n_samples
                while remaining > 0:
                    n = min(batch_size, remaining)
                    remaining -= n
                    samples = [gen.sample(rng, d, **spec.gen_kwargs)
                               for _ in range(n)]
                    if family_token_mode == "random":
                        samples = [randomize_family_token(s, rng) for s in samples]
                    batch = collate(samples, device)
                    # kwarg only when set: baseline forwards don't take it
                    kw = {"intervene": intervene} if intervene else {}
                    out = model(batch, n_iters=T, collect=True, **kw)
                    records += _records(model, out, batch, samples, spec.family, d, T)

    for r in records:
        if logger:
            logger.log(r)
    if logger:
        logger.close()
    return records


def _records(model, out, batch, samples, family, difficulty, T) -> list[dict]:
    gen = ALL_GENERATORS[family]
    tel = out.telemetry
    pred_ids = out.logits.argmax(-1)                     # [B, A]
    valid = batch["target_ids"] != -100
    B = pred_ids.shape[0]
    recs = []
    for i in range(B):
        s = samples[i]
        n_ans = int(valid[i].sum())
        pred_tokens = VOCAB.decode(pred_ids[i, :n_ans].tolist())
        rec = {
            "family": family, "difficulty": difficulty,
            "n_iters": T if T is not None else tel.get("iters_used_mean"),
            "pred": pred_tokens, "target": s.targets,
            "epistemic_true": s.epistemic,
            "requires_distinction": s.requires_distinction,
            "reconsider_label": s.reconsider_label,
            # Mathematical Generalization Surface coordinates
            "D": s.meta.get("D"), "R": s.meta.get("R"), "L": len(s.tokens),
            **gen.score(s, pred_tokens),
        }
        for key in ("iters_used", "active_slots", "expand_count",
                    "split_count", "reconsider_count", "C_max"):
            if key in tel:
                rec[key] = tel[key][i]
        if "signal_max" in tel:
            rec["signals"] = {k: v[i] for k, v in tel["signal_max"].items()}
        if "signal_max_pre" in tel:
            rec["signals_pre"] = {k: v[i]
                                  for k, v in tel["signal_max_pre"].items()}
            rec["C_max_pre"] = tel["C_max_pre"][i]
        if "state_norm" in tel:
            rec["state_norm"] = tel["state_norm"]
            rec["state_delta"] = tel["state_delta"]
        # per-iteration probe answers -> tempting-candidate trajectory and,
        # when the family has an exact checker, the residual trajectory
        if "probe_preds" in tel:
            traj_ids = tel["probe_preds"][i]                      # [T, A]
            if "attractor" in s.meta:
                attr_id = VOCAB.t(s.meta["attractor"])
                first = traj_ids[:, 0].tolist()
                rec["attractor_hit_at"] = [t for t, p in enumerate(first)
                                           if p == attr_id]
                rec["probe_traj"] = VOCAB.decode(first)
            res = [gen.residual(s, VOCAB.decode(traj_ids[t, :n_ans].tolist()))
                   for t in range(traj_ids.shape[0])]
            if all(x is not None for x in res):
                rec["residual_traj"] = res
        # slot<->variable binding diagnostics: ownership matrix if the model
        # has one, else cross-attention mass — before any claim that SPLIT
        # "separated a concept", check whether the pair's evidence even lands
        # on distinct slots
        if "pair" in s.meta:
            if "bind_matrix" in tel:
                rec["binding"] = _binding_from_B(
                    tel["bind_matrix"][i], tel["active_mask"][i], s,
                    tel.get("n_registers", 0))
            elif "cross_attn" in tel:
                rec["binding"] = _binding(tel["cross_attn"][i],
                                          tel["active_mask"][i], s)
        # Preserve an explicit zero: v0.4.52 evaluates Z0 before recurrent
        # computation and its FLOP accounting must not silently charge the
        # model's default iteration budget.
        iters = rec.get("iters_used", rec["n_iters"])
        if iters is None:
            iters = model.cfg.n_iters_train
        rec["flops"] = model.flops_per_sample(
            len(s.tokens), int(iters), n_ans=n_ans,
            active_slots=rec.get("active_slots"))
        recs.append(rec)
    return recs


def _binding(cross_attn: torch.Tensor, active: torch.Tensor, s) -> dict:
    """cross_attn [S, L] (final-iteration, mean heads) · active [S].
    Alignment(slot, variable) via attention mass over each pair member's
    token positions."""
    ta, tb = s.meta["pair"]
    pos_a = [i for i, t in enumerate(s.tokens) if t == ta]
    pos_b = [i for i, t in enumerate(s.tokens) if t == tb]
    if not pos_a or not pos_b:
        return {}
    mask = active > 0.5
    mass_a = cross_attn[:, pos_a].sum(-1).masked_fill(~mask, -1)
    mass_b = cross_attn[:, pos_b].sum(-1).masked_fill(~mask, -1)
    slot_a, slot_b = int(mass_a.argmax()), int(mass_b.argmax())
    total_a = float(cross_attn[:, pos_a].sum().clamp_min(1e-9))
    total_b = float(cross_attn[:, pos_b].sum().clamp_min(1e-9))
    return {"slot_a": slot_a, "slot_b": slot_b,
            "separated": slot_a != slot_b, "source": "attn",
            "selectivity_a": float(mass_a.max()) / total_a,
            "selectivity_b": float(mass_b.max()) / total_b}


def _binding_from_B(B_mat: torch.Tensor, active: torch.Tensor, s,
                    n_registers: int) -> dict:
    """B_mat [V, S] ownership matrix. slot_e = argmax_s B[e, s]; overlap =
    sum_s min(B_a, B_b) is the collision quantity the controller sees."""
    va, vb = (int(t[1:]) for t in s.meta["pair"])
    mask = active > 0.5
    Ba = B_mat[va].masked_fill(~mask, 0.0)
    Bb = B_mat[vb].masked_fill(~mask, 0.0)
    slot_a, slot_b = int(Ba.argmax()), int(Bb.argmax())
    return {"slot_a": slot_a, "slot_b": slot_b,
            "separated": slot_a != slot_b, "source": "bind",
            "overlap": float(torch.minimum(Ba, Bb).sum()),
            "selectivity_a": float(Ba.max() / Ba.sum().clamp_min(1e-9)),
            "selectivity_b": float(Bb.max() / Bb.sum().clamp_min(1e-9)),
            "in_registers": slot_a < n_registers or slot_b < n_registers}
