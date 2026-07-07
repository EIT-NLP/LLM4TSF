import torch
import torch.nn as nn


class TSDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_patches: int,
        pred_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.pred_len = pred_len

        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_patches * d_model, pred_len),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        num_vars: int,
    ) -> torch.Tensor:
        x = self.flatten(hidden_states)
        x = self.head(x)
        x = x.view(batch_size, num_vars, self.pred_len)
        x = x.permute(0, 2, 1).contiguous()
        return x
