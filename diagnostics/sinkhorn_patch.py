"""Scratch-only architectural prototype: capacity-limited ownership.

Repo today:   B[e,s] = softmax_s(E_e^T W_B z_s)   -- every variable picks a slot
              independently; nothing stops several pairs piling into one slot.
Prototype:    same scores, same parameters, but a few Sinkhorn-style sweeps
              alternate (rows) each variable's ownership sums to 1 with
              (columns) a slot's total ownership mass is capped at CAP.
              CAP = 2: a slot may hold one shared concept (a symmetric pair,
              the design's premise before SPLIT), not three or four variables.
No labels, no new weights, no data change.  Applied by monkeypatch so the
repository is untouched; set FAR2_SINKHORN=1 to enable.
"""

import math
import os

import torch
import torch.nn.functional as F

from far2.models.far2 import FAR2Model
from far2.models.layers import MASK_FILL

CAP = float(os.environ.get("FAR2_SINKHORN_CAP", "2"))
SWEEPS = int(os.environ.get("FAR2_SINKHORN_SWEEPS", "6"))


def _bind_matrix_capped(self, E, Z, active, ownership_bias=None):
    scores = self.bind_W(E) @ Z.transpose(1, 2) / (self.cfg.d_model ** 0.5)
    if ownership_bias is not None:
        scores = scores + ownership_bias
    logp = scores + torch.where(active.bool(), 0.0, MASK_FILL)[:, None, :]
    log_cap = math.log(CAP)
    for _ in range(SWEEPS):
        logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)      # rows -> 1
        col = torch.logsumexp(logp, dim=-2, keepdim=True)              # slot mass
        logp = logp - (col - log_cap).clamp_min(0.0)                   # cap columns
    return F.softmax(logp, dim=-1)                                     # rows sum to 1


if os.environ.get("FAR2_SINKHORN") == "1":
    FAR2Model._bind_matrix = _bind_matrix_capped
    print(f"[patch] capacity-limited ownership ON (cap={CAP}, sweeps={SWEEPS})", flush=True)
