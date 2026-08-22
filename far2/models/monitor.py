"""Collision signals (blueprint O5). The controller's hypothesis signal is
C = cos(z_i, z_j) x JSD(p_i, p_j) — "represented similarly, behaving
differently" — with alternatives computed alongside so Phase 4 can benchmark
instead of assume."""

import torch
import torch.nn.functional as F


def collision_signals(Z: torch.Tensor, active: torch.Tensor, slot_probs: torch.Tensor,
                      cross_probs: torch.Tensor | None = None) -> dict:
    """Z [B,S,d] · active [B,S] float · slot_probs [B,S,V] (per-slot answer
    distribution through the shared head) · cross_probs [B,S,L] optional
    (mean-head cross-attention maps). Returns per-pair matrices and per-sample
    summaries; invalid pairs (inactive or diagonal) are zeroed."""
    B, S, _ = Z.shape
    pair_ok = active[:, :, None] * active[:, None, :]
    pair_ok = pair_ok * (1 - torch.eye(S, device=Z.device))[None]

    zn = F.normalize(Z, dim=-1)
    sim = (zn @ zn.transpose(1, 2)).clamp(0)          # negative similarity ≠ collision

    p = slot_probs.clamp_min(1e-9)
    pi, pj = p[:, :, None, :], p[:, None, :, :]
    m = 0.5 * (pi + pj)
    jsd = 0.5 * ((pi * (pi / m).log()).sum(-1) + (pj * (pj / m).log()).sum(-1))

    out = {
        "sim": sim * pair_ok,
        "jsd": jsd * pair_ok,
        "C": (sim * jsd) * pair_ok,
    }
    if cross_probs is not None:
        ai, aj = cross_probs[:, :, None, :], cross_probs[:, None, :, :]
        iou = torch.minimum(ai, aj).sum(-1) / torch.maximum(ai, aj).sum(-1).clamp_min(1e-9)
        out["attn_iou"] = iou * pair_ok
        out["attn_iou_x_jsd"] = iou * jsd * pair_ok

    flat = out["C"].flatten(1)
    out["C_max"] = flat.max(dim=1).values                       # [B]
    argmax = flat.argmax(dim=1)
    out["C_argmax_pair"] = torch.stack([argmax // S, argmax % S], dim=1)  # [B,2]
    return out
