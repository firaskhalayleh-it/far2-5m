"""One trainer for all three models (blueprint Part VI): loss ledger, depth
sampling, param groups, EMA, JSONL logging, checkpointing."""

import json
import math
import os
import random
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from far2.config import ModelConfig, TrainConfig
from far2.data.base import collate
from far2.data.registry import MixSampler
from far2.models.base import ModelOutput
from far2.train.curriculum import Curriculum, Stage
from far2.utils import JsonlLogger, pick_device, set_seed

NO_DECAY_KEYS = ("bias", "ln", "rms", "norm", "emb", "seeds", "type_emb",
                 "goal_query", "tok", "pos")


def param_groups(model, weight_decay: float, lr: float):
    """Decay / no-decay split, with gates in their own group so optimizer
    normalization can't silently cancel their scaling."""
    decay, no_decay, gates = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if "gate" in low:
            gates.append(p)
        elif p.ndim < 2 or any(k in low for k in NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
            {"params": gates, "weight_decay": 0.0, "lr": lr}]


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1.0)))


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @contextmanager
    def swap(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


class Trainer:
    def __init__(self, model, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 stages: list[Stage] | Curriculum):
        self.model, self.mc, self.tc = model, model_cfg, train_cfg
        self.device = pick_device(train_cfg.device)
        self.model.to(self.device)
        self.curriculum = (stages if isinstance(stages, Curriculum)
                           else Curriculum(stages, train_cfg.steps))
        self.sampler = MixSampler(self.curriculum.at(0).mixes, seed=train_cfg.seed)
        self._stage_name = None
        self.opt = torch.optim.AdamW(
            param_groups(model, train_cfg.weight_decay, train_cfg.lr),
            lr=train_cfg.lr, betas=(0.9, 0.95))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_lambda(s, train_cfg.warmup, train_cfg.steps))
        self.ema = EMA(model, train_cfg.ema) if train_cfg.ema > 0 else None
        os.makedirs(train_cfg.out_dir, exist_ok=True)
        self.log = JsonlLogger(os.path.join(train_cfg.out_dir, "train_log.jsonl"))
        self.depth_rng = random.Random(train_cfg.seed + 1)
        self.train_flops = 0.0            # cumulative training-compute estimate
        self.optimizer_updates = 0
        self.unique_examples = 0
        self.forward_cycles = 0
        self.intermediate_supervised_states = 0
        with open(os.path.join(train_cfg.out_dir, "config.json"), "w") as fh:
            json.dump({"model": model_cfg.to_dict(), "train": train_cfg.to_dict(),
                       "curriculum": self.curriculum.to_dict(),
                       "params": model.param_count()}, fh, indent=2)

    # ---- depth sampling ------------------------------------------------
    def sample_depth(self, stage: Stage) -> int:
        if stage.depth == "fixed":
            return self.mc.n_iters_train
        if stage.depth == "random":
            return self.depth_rng.choice(list(self.tc.depth_choices))
        # heavy: log-normal Poisson tail (contingency ladder rung 1)
        lam = math.exp(self.depth_rng.gauss(math.log(self.mc.n_iters_train), 0.5))
        t = 1 + sum(1 for _ in range(64) if self.depth_rng.random() < lam / 64)
        return max(2, min(t, self.mc.n_iters_max))

    # ---- loss ledger ---------------------------------------------------
    def compute_loss(self, out: ModelOutput, batch, stage: Stage):
        tc = self.tc
        logits, lt = out.logits, out.loss_terms
        V = logits.shape[-1]
        targets = batch["target_ids"]
        ce_final = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                   ignore_index=-100)
        parts = {"ce": float(ce_final.detach())}
        alpha = stage.ipt_alpha or tc.ipt_alpha
        if "cycle_logits" in lt and alpha > 0:
            cycle = lt["cycle_logits"]
            ce_cycle = F.cross_entropy(
                cycle.reshape(-1, V),
                targets[:, None, :].expand(-1, cycle.shape[1], -1).reshape(-1),
                ignore_index=-100)
            loss = (1 - alpha) * ce_final + alpha * ce_cycle
            parts["ce_cycle"] = float(ce_cycle.detach())
        elif "ipt_logits" in lt and alpha > 0:
            ce_ipt = F.cross_entropy(lt["ipt_logits"].reshape(-1, V),
                                     targets.reshape(-1), ignore_index=-100)
            loss = (1 - alpha) * ce_final + alpha * ce_ipt
            parts["ce_ipt"] = float(ce_ipt.detach())
        else:
            loss = ce_final

        def add(name, weight, term):
            nonlocal loss
            if weight and isinstance(term, torch.Tensor) and term.requires_grad:
                loss = loss + weight * term.mean()
                parts[name] = float(term.detach().mean())

        add("state_sq", tc.state_norm_penalty, lt.get("state_sq"))
        add("fp_delta", tc.fp_penalty, lt.get("fp_delta"))
        add("slot_cost", stage.slot_cost or tc.slot_cost, lt.get("slot_cost"))
        add("iter_cost", stage.iter_cost or tc.iter_cost, lt.get("iter_cost"))
        bind_w = stage.bind_weight or tc.bind_weight
        add("bind_nll", bind_w, lt.get("bind_nll"))
        add("entity_nll", bind_w, lt.get("entity_nll"))
        ent_w = stage.entropy_bonus or tc.entropy_bonus
        if ent_w and lt.get("ctrl_entropy") is not None and lt["ctrl_entropy"].requires_grad:
            loss = loss - ent_w * lt["ctrl_entropy"]
            parts["ctrl_entropy"] = float(lt["ctrl_entropy"].detach())

        # Curriculum-signal supervision: requires_distinction / reconsider_label
        # are task conditions, not internal-truth oracles (see docs/DESIGN.md).
        ow = stage.oracle_weight or tc.oracle_weight
        if ow:
            for key, label in (("split_logit", "requires_distinction"),
                               ("reconsider_logit", "reconsider_label")):
                lg, lab = lt.get(key), batch[label]
                mask = lab >= 0
                if lg is not None and lg.requires_grad and mask.any():
                    bce = F.binary_cross_entropy_with_logits(
                        lg[mask], lab[mask].float())
                    loss = loss + ow * bce
                    parts[key] = float(bce.detach())

        if tc.epistemic_weight and "ans_epi_logits" in lt:
            lab = batch["epistemic_label"]
            if tc.epistemic_shuffled:
                # v0.3 control: same head, same loss pressure, zero epistemic
                # content — isolates the mechanism from aux-loss regularization
                lab = lab[torch.randperm(lab.shape[0], device=lab.device)]
            mask = lab >= 0
            if mask.any():
                ce = F.cross_entropy(lt["ans_epi_logits"][mask], lab[mask])
                loss = loss + tc.epistemic_weight * ce
                parts["epistemic"] = float(ce.detach())

        if tc.halt_q_weight and "halt_q_logit" in lt:
            with torch.no_grad():
                pred = logits.argmax(-1)
                valid = targets != -100
                correct = ((pred == targets) | ~valid).all(dim=-1).float()
            bce = F.binary_cross_entropy_with_logits(lt["halt_q_logit"], correct)
            loss = loss + tc.halt_q_weight * bce
            parts["halt_q"] = float(bce.detach())

        return loss, parts

    # ---- main loop -----------------------------------------------------
    def train(self):
        tc = self.tc
        set_seed(tc.seed)
        self.model.train()
        for step in range(tc.steps):
            stage = self.curriculum.at(step)
            if stage.name != self._stage_name:
                self._stage_name = stage.name
                self.sampler.set_mixes(stage.mixes)
                if hasattr(self.model, "set_unlocked"):
                    self.model.set_unlocked(stage.unlocked)
                    self.model.set_tau(stage.tau)
            # random Abacus offset per batch: every significance embedding
            # trains, so unseen digit lengths extrapolate (eval uses offset 0)
            batch = collate(self.sampler.batch(tc.batch_size), self.device,
                            sig_offset=self.sampler.sig_offset())
            T = self.sample_depth(stage)
            use_intermediate = (stage.ipt_alpha or tc.ipt_alpha) > 0
            mode = tc.cycle_supervision
            if mode not in ("random", "all", "final"):
                raise ValueError(f"unknown cycle_supervision={mode!r}")
            n_decodes = (T if use_intermediate and mode == "all"
                         and self.mc.kind != "transformer" else 1)
            # forward+backward ≈ 3x forward — same convention for every model,
            # so FLOP-matched comparisons (blueprint Part VI) have a denominator
            self.train_flops += 3 * tc.batch_size * self.model.flops_per_sample(
                batch["input_ids"].shape[1], T,
                n_ans=batch["target_ids"].shape[1], n_decodes=n_decodes)
            self.unique_examples += tc.batch_size
            self.forward_cycles += tc.batch_size * (T if self.mc.kind != "transformer"
                                                     else self.mc.n_layers)
            self.intermediate_supervised_states += tc.batch_size * max(n_decodes - 1, 0)
            out = self.model(
                batch, n_iters=T,
                ipt=use_intermediate and mode == "random",
                all_cycle_logits=use_intermediate and mode == "all")
            loss, parts = self.compute_loss(out, batch, stage)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            self.opt.step()
            self.optimizer_updates += 1
            self.sched.step()
            if self.ema:
                self.ema.update(self.model)

            update = step + 1
            if update in set(tc.checkpoint_steps):
                self.save_checkpoint(os.path.join(
                    tc.out_dir, f"ckpt_step{update}.pt"))

            if step % tc.log_every == 0 or step == tc.steps - 1:
                rec = {"step": step, "stage": stage.name, "T": T,
                       "loss": float(loss.detach()), "lr": self.sched.get_last_lr()[0],
                       **parts}
                for k in ("gate_mean", "active_frac", "iters_used_mean"):
                    if k in out.telemetry:
                        rec[k] = out.telemetry[k]
                sn = out.telemetry.get("state_norm")
                if sn:
                    rec["state_norm_last"] = sn[-1]
                self.log.log(rec)

        self.save()
        return self

    @contextmanager
    def eval_weights(self):
        """Prefer EMA weights for evaluation."""
        if self.ema:
            with self.ema.swap(self.model):
                yield
        else:
            yield

    def save(self):
        path = os.path.join(self.tc.out_dir, "ckpt.pt")
        self.save_checkpoint(path)
        with open(os.path.join(self.tc.out_dir, "summary.json"), "w") as fh:
            json.dump({"params": self.model.param_count(),
                       "train_flops": self.train_flops,
                       "steps": self.tc.steps,
                       "optimizer_updates": self.optimizer_updates,
                       "unique_examples": self.unique_examples,
                       "forward_cycles": self.forward_cycles,
                       "intermediate_supervised_states":
                           self.intermediate_supervised_states,
                       "intermediate_loss_weight": self.tc.ipt_alpha,
                       "cycle_supervision": self.tc.cycle_supervision},
                      fh, indent=2)

    def save_checkpoint(self, path: str):
        """Save weights plus exact compute counters at a learning-curve point."""
        torch.save({"model": self.model.state_dict(),
                    "ema": self.ema.shadow if self.ema else None,
                    "model_cfg": self.mc.to_dict(),
                    "train_cfg": self.tc.to_dict(),
                    "train_state": {
                        "train_flops": self.train_flops,
                        "optimizer_updates": self.optimizer_updates,
                        "unique_examples": self.unique_examples,
                        "forward_cycles": self.forward_cycles,
                        "intermediate_supervised_states":
                            self.intermediate_supervised_states,
                    }}, path)

    @staticmethod
    def load_model(path: str, device=None):
        from far2.models import build_model
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["model_cfg"])
        model = build_model(cfg)
        weights = ckpt["ema"] if ckpt.get("ema") else ckpt["model"]
        missing, unexpected = model.load_state_dict(weights, strict=False)
        # v0.4.4 changed the binding substrate itself.  `strict=False` is
        # retained for benign historical buffers, but a legacy bind checkpoint
        # must never masquerade as compatible while its entity detector is
        # freshly random.  Those runs must be retrained.
        critical = [k for k in missing
                    if ((cfg.kind == "far2" and k.startswith("out_pos"))
                        or (cfg.binding and (k.startswith("entity_")
                                             or k.startswith("bind_W")
                                             or k.startswith("split_dir_proj"))))]
        if critical:
            raise RuntimeError(
                "legacy FAR2 checkpoint is incompatible with the workspace-only "
                "query/contextual-entity substrate; retrain this arm (missing: "
                + ", ".join(critical[:4]) + ("..." if len(critical) > 4 else "")
                + ")")
        if unexpected:
            # Unexpected learned tensors indicate a different architecture,
            # not a safe forward-compatible buffer difference.
            raise RuntimeError("checkpoint contains unexpected model tensors: "
                               + ", ".join(unexpected[:4]))
        if device is not None:
            model.to(device)
        return model, cfg
