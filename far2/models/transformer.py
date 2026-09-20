"""Baseline A — standard pre-LN bidirectional transformer (blueprint III·A).
The no-recurrence, no-state control."""

import torch
import torch.nn as nn

from far2.config import ModelConfig
from far2.models.base import ModelOutput, ReasoningModel, gather_ans
from far2.models.layers import Block, n_params, pad_to_additive, sig_embedding


class TransformerModel(ReasoningModel):
    kind = "transformer"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.max_seq_len, d)
        self.sig = sig_embedding(d) if cfg.digit_sig else None
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)

    def forward(self, batch, n_iters=None, ipt=False, collect=False,
                all_cycle_logits=False) -> ModelOutput:
        ids = batch["input_ids"]
        B, L = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        if self.sig is not None:
            x = x + self.sig(batch["digit_sig"])
        mask = pad_to_additive(batch["pad_mask"])
        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.head(gather_ans(self.ln_f(x), batch["ans_pos"]))
        return ModelOutput(logits, telemetry={"iters_used_mean": float(self.cfg.n_layers)})

    def flops_per_sample(self, seq_len, n_iters=None, n_ans=1, active_slots=None,
                         n_decodes=1):
        p_block = n_params(self.blocks[0])
        d = self.cfg.d_model
        per_block = 2 * p_block * seq_len + 4 * seq_len ** 2 * d
        return self.cfg.n_layers * per_block + 2 * d * self.cfg.vocab_size * n_ans
