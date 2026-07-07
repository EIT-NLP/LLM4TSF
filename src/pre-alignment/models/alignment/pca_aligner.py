import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAWordAligner(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.residual = residual

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ts_embeddings: torch.Tensor,
        pca_word_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if pca_word_embeddings.dim() != 2:
            raise ValueError(
                f"Expected pca_word_embeddings with shape [K, d_model], "
                f"but got {pca_word_embeddings.shape}"
            )

        batch_size = ts_embeddings.size(0)
        dictionary = pca_word_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        aligned, _ = self.attn(
            query=ts_embeddings,
            key=dictionary,
            value=dictionary,
            need_weights=False,
        )

        aligned = self.dropout(aligned)

        if self.residual:
            aligned = self.norm(ts_embeddings + aligned)
        else:
            aligned = self.norm(aligned)

        return aligned


def build_pca_word_embeddings(
    word_embeddings: torch.Tensor,
    num_components: int,
) -> torch.Tensor:
    if word_embeddings.dim() != 2:
        raise ValueError(
            f"Expected word_embeddings with shape [vocab_size, d_model], "
            f"but got {word_embeddings.shape}"
        )

    vocab_size, _ = word_embeddings.shape

    if num_components > vocab_size:
        raise ValueError(
            f"num_components={num_components} is larger than vocab_size={vocab_size}"
        )

    with torch.no_grad():
        x = word_embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        pca_embeddings = vh[:num_components]
        pca_embeddings = F.normalize(pca_embeddings, dim=-1)

    return pca_embeddings