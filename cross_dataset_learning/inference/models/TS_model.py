import torch
import torch.nn as nn
from torch import cosine_similarity
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

    def register_similarity_hooks_gpt(self):
        ### sublayer
        def hook_factory(layer_name):
            def hook(module, input, output):
                try:
                    inp = input[0].detach()
                    out = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
                    sim = cosine_similarity(inp, out, dim=-1).mean().item()
                    dist = torch.norm(inp - out, dim=-1).mean().item()
                    self.sub_similarities[layer_name].append(sim)
                    self.sub_dist[layer_name].append(dist)
                except Exception as e:
                    print(f"[Hook Error] {layer_name}: {e}")
            return hook

        self.sub_similarities = {}
        self.sub_dist = {}

        for i, block in enumerate(self.gpt2.h):
            for sub_name in ['ln_1', 'attn', 'ln_2', 'mlp']:
                module = getattr(block, sub_name, None)
                if module is not None:
                    layer_name = f"h.{i}.{sub_name}"
                    self.sub_similarities[layer_name] = []
                    self.sub_dist[layer_name] = []
                    module.register_forward_hook(hook_factory(layer_name))

        if hasattr(self.gpt2, 'ln_f'):
            self.sub_similarities['ln_f'] = []
            self.sub_dist['ln_f'] = []
            self.gpt2.ln_f.register_forward_hook(hook_factory('ln_f'))

    def register_block_similarity_hooks(self):
        ### block
        def hook_factory(layer_name):
            def hook(module, input, output):
                try:
                    inp = input[0].detach()
                    out = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
                    dist = torch.norm(inp - out, dim=-1).mean().item()
                    sim = cosine_similarity(inp, out, dim=-1).mean().item()
                    self.block_similarities[layer_name].append(sim)
                    self.block_dist[layer_name].append(dist)
                except Exception as e:
                    print(f"[Hook Error] {layer_name}: {e}")
            return hook

        self.block_similarities = {}
        self.block_dist = {}

        for i, block in enumerate(self.gpt2.h):
            layer_name = f"block_{i}"
            self.block_similarities[layer_name] = []
            self.block_dist[layer_name] = []
            block.register_forward_hook(hook_factory(layer_name))

        if hasattr(self.gpt2, 'ln_f'):
            self.block_similarities["ln_f"] = []
            self.block_dist["ln_f"] = []
            self.gpt2.ln_f.register_forward_hook(hook_factory("ln_f"))

    def save_all_attentions(self, save_path):
        saved_data = {
            'config': {
                'num_batches': len(self.all_attentions),
                'num_layers': len(self.all_attentions[0]) if self.all_attentions else 0,
            },
            'all_attentions': [
                [attn.cpu().float() for attn in att_per_batch]
                for att_per_batch in self.all_attentions
            ]
        }
        torch.save(saved_data, save_path)

class BlockSimilarityTracker:
    def __init__(self, configs, model):
        self.model = model
        self.layer_names = [f"block_{i}" for i in range(len(model.h))]
        if hasattr(model, "ln_f"):
            self.layer_names.append("ln_f")

        self.num_layers = len(self.layer_names) * 2  # input + output
        self.similarity_sum = torch.zeros((self.num_layers, self.num_layers))
        self.dist_sum = torch.zeros((self.num_layers, self.num_layers))
        self.batch_count = 0
        if configs.heatmap:
            self._register_hooks()

    def _register_hooks(self):
        def hook_factory(layer_name):
            def hook(module, input, output):
                inp = input[0].detach().cpu()
                out = output.detach().cpu() if isinstance(output, torch.Tensor) else output[0].detach().cpu()

                self.current_batch[layer_name] = (inp, out)
            return hook

        self.current_batch = {}

        for i, block in enumerate(self.model.h):
            block.register_forward_hook(hook_factory(f"block_{i}"))
        if hasattr(self.model, "ln_f"):
            self.model.ln_f.register_forward_hook(hook_factory("ln_f"))

    def step(self):
        all_feats = []
        for name in self.layer_names:
            inp, out = self.current_batch[name]  # [B, P, D]
            all_feats.append(inp.reshape(-1, inp.shape[-1]))   # flatten to [B*P, D]
            all_feats.append(out.reshape(-1, out.shape[-1]))

        sim_matrix = torch.zeros((self.num_layers, self.num_layers))
        dist_matrix = torch.zeros((self.num_layers, self.num_layers))
        for i in range(self.num_layers):
            for j in range(self.num_layers):
                sim_matrix[i, j] = cosine_similarity(all_feats[i], all_feats[j], dim=-1).mean()
                dist_matrix[i, j] = torch.norm(all_feats[i]-all_feats[j], dim=-1).mean()
        self.similarity_sum += sim_matrix
        self.dist_sum += dist_matrix
        self.batch_count += 1
        self.current_batch.clear()
