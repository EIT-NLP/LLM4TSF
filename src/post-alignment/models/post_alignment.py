import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .patch_embedding import PatchEmbedding
from .ts_encoder import TSEncoder
from .ts_decoder import TSDecoder


class PostAlignmentLLM4TSF(nn.Module):
    def __init__(
        self,
        llm_name: str,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 768,
        dropout: float = 0.1,
        freeze_llm: bool = False,
        use_prompt: bool = True,
        prompt_text: str = "Forecast the future values of the given time series.",
    ):
        super().__init__()

        self.llm_name = llm_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.freeze_llm = freeze_llm
        self.use_prompt = use_prompt
        self.prompt_text = prompt_text

        self.num_patches = (seq_len - patch_len) // stride + 2

        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
        )

        self.ts_encoder = TSEncoder(
            patch_len=patch_len,
            d_model=d_model,
            dropout=dropout,
        )

        self.llm = AutoModel.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_hidden_size = self.llm.config.hidden_size

        if d_model != llm_hidden_size:
            self.ts_to_llm = nn.Linear(d_model, llm_hidden_size)
            self.llm_to_ts = nn.Linear(llm_hidden_size, d_model)
        else:
            self.ts_to_llm = nn.Identity()
            self.llm_to_ts = nn.Identity()

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.ts_decoder = TSDecoder(
            d_model=d_model,
            num_patches=self.num_patches,
            pred_len=pred_len,
            dropout=dropout,
        )

    def normalize(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True).detach()
        std = torch.clamp(std, min=1e-5)
        x = (x - mean) / std
        return x, mean, std

    def denormalize(
        self,
        forecast: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        return forecast * std + mean

    def build_prompt_embeddings(
        self,
        batch_size: int,
        num_vars: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompts = [self.prompt_text] * (batch_size * num_vars)

        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_ids = tokens["input_ids"]
        prompt_embeddings = self.llm.get_input_embeddings()(input_ids)
        return prompt_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, L, C], but got {x.shape}")

        batch_size, seq_len, num_vars = x.shape

        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected input seq_len={self.seq_len}, but got seq_len={seq_len}"
            )

        x, mean, std = self.normalize(x)

        patches = self.patch_embedding(x)
        ts_embeddings = self.ts_encoder(patches)
        ts_embeddings = self.ts_to_llm(ts_embeddings)

        if self.use_prompt:
            prompt_embeddings = self.build_prompt_embeddings(
                batch_size=batch_size,
                num_vars=num_vars,
                device=x.device,
            )
            llm_inputs = torch.cat([prompt_embeddings, ts_embeddings], dim=1)
            prompt_len = prompt_embeddings.size(1)
        else:
            llm_inputs = ts_embeddings
            prompt_len = 0

        llm_outputs = self.llm(inputs_embeds=llm_inputs)
        hidden_states = llm_outputs.last_hidden_state
        ts_hidden = hidden_states[:, prompt_len:, :]
        ts_hidden = self.llm_to_ts(ts_hidden)

        forecast = self.ts_decoder(
            hidden_states=ts_hidden,
            batch_size=batch_size,
            num_vars=num_vars,
        )

        forecast = self.denormalize(forecast, mean, std)
        return forecast
