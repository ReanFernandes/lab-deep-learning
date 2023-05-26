import torch
import torch.nn as nn
from torch.nn import functional as F
from operations_model import *
from operations.mixed_embedding import MixedEmbedding, MixedEmbeddingV2
from operations.mixed_block import MixedBlock
from operations.mixed_layer_norm import MixedLayerNorm, MixedLayerNormV2
from operations.mixed_linear import MixedLinearHead, MixedLinearHeadV2
from optimizers.optim_factory import get_mixop, get_sampler
from optimizers.mixop.entangle import EntangledOp
import itertools
import numpy as np


class BigramLanguageModel(nn.Module):

    def __init__(self, choices={}, vocab_size=10000, block_size=256, dropout=0.0, device='cuda', mixop="darts_v1", use_we_v2=False):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # Note attention does not have any notion of colocation of characters/words and this is important for lms
        self.vocab_size = vocab_size
        self.max_embed_dim = max(choices["embed_dim"])
        self.max_num_layers = max(choices["num_layers"])
        self.choices = choices
        self.block_size = block_size
        self.dropout = dropout
        self.device = device
        self.num_layers = None
        self.mixop = get_mixop(mixop, use_we_v2=use_we_v2)
        self.sampler = get_sampler(mixop)
        self.token_embedding_table = nn.Embedding(
            vocab_size, self.max_embed_dim)
        if use_we_v2:
            self.token_embedding_table_op = MixedEmbeddingV2(
                choices["embed_dim"], max_embed_dim=self.max_embed_dim, embedding=self.token_embedding_table)
            self.token_embedding_table_list = self.get_entangle_ops(
                self.token_embedding_table_op, choices["embed_dim"], "embedding_table")
        else:
            self.token_embedding_table_list = [MixedEmbedding(
                e, max_embed_dim=self.max_embed_dim) for e in choices["embed_dim"]]
        self.position_embedding_table = nn.Embedding(
            block_size, self.max_embed_dim)
        if use_we_v2:
            self.position_embedding_table_op = MixedEmbeddingV2(
                choices["embed_dim"], max_embed_dim=self.max_embed_dim, embedding=self.position_embedding_table)
            self.position_embedding_table_list = self.get_entangle_ops(
                self.position_embedding_table_op, choices["embed_dim"], "position_embedding_table")
        else:
            self.position_embedding_table_list = [MixedEmbedding(
                e, max_embed_dim=self.max_embed_dim) for e in choices["embed_dim"]]
        self.blocks = nn.Sequential(
            *[MixedBlock(choices, i, block_size, dropout, self.mixop, use_we_v2=use_we_v2) for i in range(max(choices["num_layers"]))])
        self.ln_f = nn.LayerNorm(self.max_embed_dim)  # final layer norm
        if use_we_v2:
            self.ln_f_op = MixedLayerNormV2(
                choices["embed_dim"], self.max_embed_dim, self.ln_f)
            self.ln_f_list = self.get_entangle_ops(
                self.ln_f_op, choices["embed_dim"], "ln_f")
        else:
            self.ln_f_list = [MixedLayerNorm(
                e, self.max_embed_dim) for e in choices["embed_dim"]]
        self.lm_head = nn.Linear(self.max_embed_dim, vocab_size)
        if use_we_v2:
            self.lm_head_op = MixedLinearHeadV2(
                choices["embed_dim"],  self.max_embed_dim, self.lm_head)
            self.lm_head_list = self.get_entangle_ops(
                self.lm_head_op, choices["embed_dim"], "lm_head")
        else:
            self.lm_head_list = [MixedLinearHead(
                e,  self.max_embed_dim) for e in choices["embed_dim"]]

        self._init_arch_parameters()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_weights_layernorm(self, choices):
        embed_choice = choices['embed_dim']
        return self.ln_f.weight[:embed_choice], self.ln_f.bias[:embed_choice]

    def get_weights_embedding(self, choices):
        embed_choice = choices['embed_dim']
        return self.token_embedding_table.weight[:, :embed_choice], self.position_embedding_table.weight[:, :embed_choice]

    def get_weights_lm_head(self, choices):
        embed_choice = choices['embed_dim']
        bias_lm_head = choices["bias_lm_head"]
        if bias_lm_head:
            return self.lm_head.weight[:, :embed_choice], self.lm_head.bias[:]
        else:
            return self.lm_head.weight[:, :embed_choice], None

    def sample_random_config(self):
        '''
        TODO: sample a random config for the model corresponding to choice of num_layers, num_heads, embed_dims, mlp_ratio
        Initialize the arch_param_dict with the sampled values. The one-hot encoded arch parameters must correspond to the sampled config
        Args: 
        Input: None
        Output: config: dict containing the sampled config
                arch_param_dict: dict containing the one-hot encoded arch parameters
        
        '''
        num_layers = self.choices["num_layers"]
        num_heads = self.choices["num_heads"]
        embed_dims = self.choices["embed_dim"]
        mlp_ratio = self.choices["mlp_ratio"]
        arch_param_dict = {}
        config = {}
        arch_param_dict["num_layers"] = self.sampler.sample(num_layers)
        config["num_layers"] = num_layers[arch_param_dict["num_layers"].argmax()]
        self.num_layers = config["num_layers"]
        arch_param_dict["embed_dim"] = self.sampler.sample(embed_dims)
        config["embed_dim"] = embed_dims[arch_param_dict["embed_dim"].argmax()]
        arch_param_dict["num_heads"] = self.sampler.sample([num_heads for _ in range(config["num_layers"])])
        config["num_heads"] = [num_heads[arch_param_dict["num_heads"][i].argmax()] for i in range(config["num_layers"])]
        arch_param_dict["mlp_ratio"] = self.sampler.sample([mlp_ratio for _ in range(config["num_layers"])])
        config["mlp_ratio"] = [mlp_ratio[arch_param_dict["mlp_ratio"][i].argmax()] for i in range(config["num_layers"])]


        return config, arch_param_dict

    def _init_arch_parameters(self):
        self.arch_parameter_dict = nn.ParameterDict()
        self.arch_parameter_dict["num_layers"] = nn.Parameter(
            1e-3 * torch.randn([len(self.choices["num_layers"])]), requires_grad=True)
        self.arch_parameter_dict["embed_dim"] = nn.Parameter(
            1e-3 * torch.randn([len(self.choices["embed_dim"])]), requires_grad=True)
        self.arch_parameter_dict["num_heads"] = nn.Parameter(
            1e-3 * torch.randn([max(self.choices["num_layers"]), len(self.choices["num_heads"])]), requires_grad=True)
        self.arch_parameter_dict["mlp_ratio"] = nn.Parameter(
            1e-3 * torch.randn([max(self.choices["num_layers"]), len(self.choices["mlp_ratio"])]), requires_grad=True)

    def get_arch_parameters(self):
        return list(self.arch_parameter_dict.values())

    def get_model_parameters(self):
        param_list = []
        for name, param in self.named_parameters():
            if "arch_parameter_dict" not in name:
                param_list.append(param)
        return param_list

    def assign_arch_parameters(self, arch_parameters):
        arch_params_dummy = {}
        for i, k in enumerate(self.arch_parameter_dict.keys()):
            if isinstance(arch_parameters, list):
                arch_params_dummy[k] = arch_parameters[i]
            else:
                arch_params_dummy[k] = arch_parameters[k]
        return arch_params_dummy

    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def forward(self, idx, targets=None, arch_params=None):
        if arch_params is None:
            arch_parameters = self.sampler.sample_step(
                self.get_arch_parameters())
            arch_params_sampled_dict = self.assign_arch_parameters(
                arch_parameters)
        else:
            arch_params_sampled_dict = self.assign_arch_parameters(arch_params)
        B, T = idx.shape
        # print(choices)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.mixop.forward_layer(
            idx, arch_params_sampled_dict["embed_dim"], self.token_embedding_table_list, self.token_embedding_table)
        # pos_emb = torch.nn.functional.embedding(torch.arange(
        #    T).to(idx.device), position_embedding_weight)  # (T,C)
        pos_emb = self.mixop.forward_layer(
            torch.arange(T).to(idx.device), arch_params_sampled_dict["embed_dim"], self.position_embedding_table_list, self.position_embedding_table)
        x = tok_emb + pos_emb  # (B,T,C)
        depth_output_list = []
        for i in range(self.num_layers):
            x = self.blocks[i](x, i, arch_params_sampled_dict)
            if i+1 in self.choices["num_layers"]:
                depth_output_list.append(x)
        x = self.mixop.forward_depth(
            depth_output_list, arch_params_sampled_dict["num_layers"])
        x = self.mixop.forward_layer(
            x, arch_params_sampled_dict["embed_dim"], self.ln_f_list, self.ln_f)
        logits = self.mixop.forward_layer(
            x, arch_params_sampled_dict["embed_dim"], self.lm_head_list, self.lm_head)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Test model
# choices = {}
# choices["num_layers"] = [1, 2, 3]
# choices["embed_dim"] = [128, 256, 512]
# choices["num_heads"] = [2, 4, 8]
# choices["mlp_ratio"] = [1, 2, 4]
# model = BigramLanguageModel(choices=choices, block_size=128, use_we_v2=False, mixop="spos")
# # initialize random input indices
# model.sampler.set_taus(0.1,10)
# model.sampler.set_total_epochs(100)
# model.sampler.before_epoch()
# idx = torch.randint(0, 100, (32, 128))
# # Initialize shifted target
# targets = torch.cat((idx[:, 1:], idx[:, 0:1]), dim=1)
# # forward pass
# logits, loss = model(idx, targets=targets)
# print(logits.shape, loss)
