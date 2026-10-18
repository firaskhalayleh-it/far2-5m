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


def rope_tables(length: int, head_dim: int, device, theta: float = 10000.0):
    """Rotary position tables [L, head_dim/2]. Relative offsets become a
    rotation, so "attend to my right neighbour" is one pattern at every
    position instead of one learned pattern per absolute position."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(length, device=device).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    cos, sin = cos[: x.shape[-2]].to(x.dtype), sin[: x.shape[-2]].to(x.dtype)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, qk_norm: bool = False,
                 qk_scale_init: float = 10.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        # QK-norm: scores become scale_h * cos(q, k).  The logit range is a
        # learned per-head temperature instead of a slow growth of |Wq||Wk|,
        # so attention can become sharp early (1/178 -> one token needs a
        # logit gap of ~5; LN'd inputs at init give score std ~0.33).
        self.qk_scale = (nn.Parameter(torch.full((n_heads,), qk_scale_init).log())
                         if qk_norm else None)

    def forward(self, q_in, kv_in, mask=None, need_probs: bool = False,
                competitive: bool = False, slot_mask=None, rope=None):
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
        if rope is not None:                       # self-attention only
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if self.qk_scale is not None:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            scores = (q @ k.transpose(-2, -1)) * self.qk_scale.exp()[None, :, None, None]
        else:
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

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, act: str = "gelu",
                 qk_norm: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, qk_norm=qk_norm)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult, act)

    def forward(self, x, mask=None, rope=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, mask, rope=rope)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class LocalMix(nn.Module):
    """Give every token learned projections of its immediate left and right
    neighbours (Canon-style local mixing).  Relation rows such as "X4 GEQ X5"
    are then visible to the token's own MLP without any attention: X4 can form
    "X4 and my right neighbour is GEQ".  Zero-initialized, so it starts as the
    identity; padding contributes nothing."""

    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.left = nn.Linear(d_model, d_model, bias=False)
        self.right = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.left.weight)
        nn.init.zeros_(self.right.weight)

    def forward(self, x, pad_mask):
        h = self.ln(x) * pad_mask[..., None]
        prev = F.pad(h, (0, 0, 1, 0))[:, :-1]      # token k-1 at position k
        nxt = F.pad(h, (0, 0, 0, 1))[:, 1:]        # token k+1 at position k
        return x + (self.left(prev) + self.right(nxt)) * pad_mask[..., None]


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
