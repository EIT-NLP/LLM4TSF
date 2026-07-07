import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, L, C], but got {x.shape}")

        bsz, seq_len, num_vars = x.shape
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.pad(x, (0, self.stride), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(bsz * num_vars, patches.size(2), self.patch_len)
        return patches
