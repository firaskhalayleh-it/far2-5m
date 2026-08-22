"""Model interface shared by all three architectures (blueprint Part III).
forward() consumes the collate() batch and returns logits at ANS positions plus
loss terms the trainer weights and telemetry the harness records."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    logits: torch.Tensor                      # [B, A, vocab]
    loss_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


class ReasoningModel(nn.Module):
    kind: str

    @abstractmethod
    def forward(self, batch: dict, n_iters: int | None = None,
                ipt: bool = False, collect: bool = False) -> ModelOutput: ...

    @abstractmethod
    def flops_per_sample(self, seq_len: int, n_iters: int,
                         n_ans: int = 1, active_slots: float | None = None) -> float: ...

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def gather_ans(hidden: torch.Tensor, ans_pos: torch.Tensor) -> torch.Tensor:
    """hidden [B,L,d], ans_pos [B,A] (-1 padded) -> [B,A,d]. Padded positions
    gather index 0; their targets are -100 so the loss ignores them."""
    idx = ans_pos.clamp_min(0)[..., None].expand(-1, -1, hidden.shape[-1])
    return hidden.gather(1, idx)
