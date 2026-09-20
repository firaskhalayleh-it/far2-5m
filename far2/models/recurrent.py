"""Baseline B — one shared block applied T times to the token stream
(blueprint III·B). The model FAR² must beat.

Stability recipe (docs/DESIGN.md): evidence embeddings are re-injected every
iteration (recall) into a zero-initialized state, and the block output passes
through an external RMSNorm — the placement that keeps T >> T_train bounded.
Timestep embeddings default off. IPT is fused: a random number of leading
iterations runs detached, so gradients see every window in expectation and the
model cannot learn iteration-indexed behavior."""

import random

import torch
import torch.nn as nn

from far2.config import ModelConfig
from far2.models.base import ModelOutput, ReasoningModel, gather_ans
from far2.models.layers import Block, n_params, pad_to_additive, sig_embedding


class RecurrentModel(ReasoningModel):
    kind = "recurrent"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.max_seq_len, d)
        self.sig = sig_embedding(d) if cfg.digit_sig else None
        self.block = Block(d, cfg.n_heads, cfg.mlp_mult)
        self.ln_in = nn.LayerNorm(d)
        self.rms_out = nn.RMSNorm(d)
        self.iter_emb = (nn.Embedding(cfg.n_iters_max, d)
                         if cfg.iter_embedding else None)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)

    def forward(self, batch, n_iters=None, ipt=False, collect=False,
                all_cycle_logits=False) -> ModelOutput:
        cfg = self.cfg
        ids = batch["input_ids"]
        B, L = ids.shape
        e = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        if self.sig is not None:
            e = e + self.sig(batch["digit_sig"])
        mask = pad_to_additive(batch["pad_mask"])
        T = min(n_iters or cfg.n_iters_train, cfg.n_iters_max)

        detach_after = -1
        ipt_at = -1
        if ipt and self.training and T > 1:
            detach_after = random.randint(0, T - 1)          # dirty detached start
            ipt_at = random.randint(detach_after + 1, T)     # supervised iterate

        s = torch.zeros_like(e)
        norms, deltas = [], []
        state_sq = e.new_zeros(())
        ipt_logits = None
        cycle_logits = []
        last_delta_sq = None
        for t in range(T):
            prev = s
            x = self.ln_in(s) + e                             # recall every iteration
            if self.iter_emb is not None:
                x = x + self.iter_emb.weight[t % self.cfg.n_iters_train][None, None]
            s = self.rms_out(self.block(x, mask))
            if t + 1 == detach_after:
                s = s.detach()
            state_sq = state_sq + s.pow(2).mean()
            last_delta_sq = (s - prev).pow(2).mean()
            norms.append(float(s.detach().pow(2).mean().sqrt()))
            deltas.append(float(last_delta_sq.detach().sqrt()))
            if t + 1 == ipt_at and ipt_at != T:
                ipt_logits = self.head(gather_ans(self.ln_f(s), batch["ans_pos"]))
            if all_cycle_logits and t + 1 < T:
                cycle_logits.append(
                    self.head(gather_ans(self.ln_f(s), batch["ans_pos"])))

        logits = self.head(gather_ans(self.ln_f(s), batch["ans_pos"]))
        loss_terms = {"state_sq": state_sq / T, "fp_delta": last_delta_sq}
        if ipt_logits is not None:
            loss_terms["ipt_logits"] = ipt_logits
        if cycle_logits:
            loss_terms["cycle_logits"] = torch.stack(cycle_logits, dim=1)
        return ModelOutput(logits, loss_terms, telemetry={
            "iters_used_mean": float(T), "state_norm": norms, "state_delta": deltas})

    def flops_per_sample(self, seq_len, n_iters, n_ans=1, active_slots=None,
                         n_decodes=1):
        p_block = n_params(self.block)
        d = self.cfg.d_model
        per_iter = 2 * p_block * seq_len + 4 * seq_len ** 2 * d
        return (n_iters * per_iter
                + n_decodes * 2 * d * self.cfg.vocab_size * n_ans)
