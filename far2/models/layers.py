"""Shared building blocks. Attention computes probabilities explicitly (the
telemetry needs entropies and cross-attention maps); at 5M scale the fused
kernel would buy nothing worth losing that visibility."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_FILL = -1e4  # finite: an all-masked softmax row must not produce NaN


def pad_to_additive(pad_mask: torch.Tensor) -> torch.Tensor:
    """[B, L] bool (True = real) -> [B, 1, 1, L] additive float mask."""
    return torch.where(pad_mask, 0.0, MASK_FILL)[:, None, None, :]


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q_in, kv_in, mask=None, need_probs: bool = False,
                competitive: bool = False, slot_mask=None):
        """q_in [B,Lq,d], kv_in [B,Lk,d], mask additive [B,1|h,1|Lq,Lk].

        competitive=True inverts the normalization axis (Slot-Attention bias):
        each EVIDENCE token softmaxes over the query slots — "which slot owns
        me?" — then each slot aggregates its winnings, renormalized. slot_mask
        is an additive [B,1,Lq,1] activity mask so inactive slots cannot
        compete. Returned probs are per-slot distributions over Lk in both
        modes, so telemetry consumers need no branching."""
        B, Lq, _ = q_in.shape
        Lk = kv_in.shape[1]
        q = self.wq(q_in).view(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.wk(kv_in).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.wv(kv_in).view(B, Lk, self.h, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if competitive:
            if slot_mask is not None:
                scores = scores + slot_mask
            A = F.softmax(scores, dim=-2)              # over slots, per token
            if mask is not None:
                A = A * (mask > -1.0).float()          # drop padded evidence
            probs = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
        else:
            if mask is not None:
                scores = scores + mask
            probs = F.softmax(scores, dim=-1)
        out = (probs @ v).transpose(1, 2).reshape(B, Lq, self.h * self.dh)
        return self.wo(out), (probs if need_probs else None)


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int, act: str = "gelu",
                 d_in: int | None = None):
        super().__init__()
        self.up = nn.Linear(d_in or d_model, mult * d_model)
        self.down = nn.Linear(mult * d_model, d_model)
        self.act = {"gelu": F.gelu, "relu": F.relu, "elu": F.elu}[act]

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class Block(nn.Module):
    """Pre-LN transformer block (self-attention + MLP)."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, act: str = "gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult, act)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


def sig_embedding(d_model: int) -> nn.Embedding:
    """Shared Abacus-style digit-significance embedding (index 0 = not a
    digit, zero vector). Identical module in ALL models — the numeric
    representation is part of the comparability contract."""
    from far2.data.vocab import MAX_SIG
    emb = nn.Embedding(MAX_SIG + 1, d_model, padding_idx=0)
    nn.init.normal_(emb.weight, std=0.02)
    with torch.no_grad():
        emb.weight[0].zero_()
    return emb


def attn_entropy(probs: torch.Tensor) -> float:
    """Mean per-row entropy of attention probabilities [B,h,Lq,Lk]."""
    p = probs.detach().clamp_min(1e-9)
    return float((-(p * p.log()).sum(-1)).mean())


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
