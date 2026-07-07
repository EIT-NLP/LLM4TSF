import torch
import torch.nn as nn


class TSEncoder(nn.Module):
    def __init__(
        self,
        patch_len: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.proj(patches)
        x = self.norm(x)
        x = self.dropout(x)
        return x
