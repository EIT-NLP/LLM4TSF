import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
class TS_model(nn.Module):
    def __init__(self, configs):
        super(TS_model, self).__init__()
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.pred_len = configs.pred_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.model_name = configs.model_name
        self.family = configs.family
        self.scales = configs.scales

        if self.family == "LLM4TS":
            if self.scales == "Tiny" or self.scales == "Small":
                self.tsmodel = GPT2Model.from_pretrained('gpt2',
                                                      attn_implementation="eager",
                                                      output_hidden_states=True)
                for name, param in self.tsmodel.named_parameters():
                    param.requires_grad = False
                    if "wpe" in name or "ln" in name:
                        param.requires_grad = True
                if self.scales == "Tiny":
                    self.tsmodel.h = self.tsmodel.h[:6]
            else:
                self.tsmodel = AutoModelForCausalLM.from_pretrained(
                    f'Qwen/{self.model_name}',
                    output_hidden_states = True,
                    attn_implementation="eager",
                )
                for name, param in self.tsmodel.named_parameters():
                    param.requires_grad = False
                    if "norm" in name or "embed_tokens" in name:
                        param.requires_grad = True
        elif self.family == "TSFMs":
            if self.scales == "Tiny" or self.scales == "Small":
                self.tsmodel = GPT2Model(GPT2Config())
                if self.scales == "Tiny":
                    self.tsmodel.h = self.tsmodel.h[:6]
            else:
                config = AutoConfig.from_pretrained(
                    f'Qwen/{self.model_name}',
                    output_hidden_states = True,
                    attn_implementation="eager",)
                self.tsmodel = AutoModelForCausalLM.from_config(config)

        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

    def forward(self, x):
        B, L, M = x.shape

        # Normalize: subtract mean and divide by std
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        # Rearrange and prepare patches
        x = rearrange(x, 'b l m -> b m l').float()
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # (b, m, n_patches, patch_size)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # Embed patches
        x_embedding = self.in_layer(x)  # [B*M, N, D]

        # Forward through tsmodel
        tsmodel_outputs = self.tsmodel(inputs_embeds=x_embedding, output_hidden_states=True)
        outputs = tsmodel_outputs.hidden_states[-1]  # [B*M, N, D]

        # Project and reshape back
        outputs = outputs.reshape(B * M, -1)  # flatten sequence and feature dims
        outputs = self.out_layer(outputs)     # [B*M, L_out]
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        # Denormalize
        outputs = outputs * stdev + means
        return outputs