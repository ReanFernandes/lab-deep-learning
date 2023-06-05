import torch.nn as nn
import torch
from optimizers.mixop.entangle import EntangledOp
import itertools
from .mixed_linear import MixedLinear, MixedLinearV2
class MixedFeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, choices, dropout, mixop, use_we_v2=False):
        super().__init__()
        self.max_embd = max(choices["embed_dim"])
        self.max_num_heads = max(choices["num_heads"])
        self.max_mlp_ratio = max(choices["mlp_ratio"])
        self.max_expansion_dim = self.max_embd * self.max_mlp_ratio
        self.mixop = mixop
        self.net = nn.Sequential(
            nn.Linear(self.max_embd, self.max_expansion_dim),
            nn.Linear(self.max_expansion_dim, self.max_embd),
            nn.Dropout(dropout),
        )
        if use_we_v2:
            self.linear_0_mix_op = MixedLinearV2(choices["embed_dim"], choices["mlp_ratio"], self.max_embd, linear_layer=self.net[0])
            self.linear_0_mixed = self.get_entangle_ops_combi(self.linear_0_mix_op, choices["embed_dim"], choices["mlp_ratio"], "linear_0_embed_dim_ff")
            self.linear_1_mix_op = MixedLinearV2(choices["embed_dim"], choices["mlp_ratio"], self.max_embd, linear_layer=self.net[1], reverse=True)
            self.linear_1_mixed = self.get_entangle_ops_combi(self.linear_1_mix_op, choices["embed_dim"], choices["mlp_ratio"], "linear_1_embed_dim_ff")
        else:
            self.linear_0_mixed = []
            for emb in choices["embed_dim"]:
                for r in choices["mlp_ratio"]:
                    self.linear_0_mixed.append(MixedLinear(emb, emb*r, self.max_expansion_dim))
            self.linear_1_mixed = []
            for emb in choices["embed_dim"]:
                for r in choices["mlp_ratio"]:
                    self.linear_1_mixed.append(MixedLinear(emb*r, emb, self.max_embd))
        self.activation = nn.ReLU()

    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]
    
    def forward(self, x, i, arch_params):
        x = self.mixop.forward_layer(x, [arch_params["embed_dim"], arch_params["mlp_ratio"][i]], self.linear_0_mixed, self.net[0], combi=True)
        x = self.activation(x)
        x = self.mixop.forward_layer(x, [arch_params["embed_dim"], arch_params["mlp_ratio"][i]], self.linear_1_mixed, self.net[1], combi=True)
        x = self.net[2](x)
        return x
    
