import torch.nn as nn
import torch
from operations.mixed_head import Head, MixedHeads, MixedHeadsV2
from operations.mixed_linear import MixedLinear, MixedLinearV2, MixedLinearV2Emb
from optimizers.mixop.entangle import EntangledOp
import itertools

class MixedMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, choices, block_size, dropout, mixop, use_we_v2=False):
        super().__init__()
        self.choices = choices
        self.max_embed_dim = max(choices["embed_dim"])
        self.max_num_heads = max(choices["num_heads"])
        self.min_embed_dim = min(choices["embed_dim"])
        self.mixop = mixop
        self.max_head_dim = self.max_embed_dim // self.max_num_heads
        self.heads = nn.ModuleList([Head(self.max_embed_dim, block_size, dropout, self.max_head_dim)
                                   for _ in range(self.max_num_heads)])
        #f use_we_v2:
        #    self.heads_mix_op = MixedHeadsV2(choices["embed_dim"], choices["num_heads"], self.max_embed_dim, base_heads=self.heads)
        #    self.heads_mixed = self.get_entangle_ops_combi(self.heads_mix_op, choices["embed_dim"], choices["num_heads"], "heads_embed_dim_msa")
        #else:
        self.heads_mixed = nn.ModuleList([])
        for embed_dim in choices["embed_dim"]:
            for num_heads in choices["num_heads"]:
                self.heads_mixed.append(MixedHeads(embed_dim, num_heads, self.max_embed_dim))
        self.proj = nn.Linear(self.max_embed_dim, self.max_embed_dim)
        if use_we_v2:
            self.proj_mix_op = MixedLinearV2Emb(choices["embed_dim"], self.max_embed_dim, linear_layer=self.proj)
            self.proj_mixed = self.get_entangle_ops(self.proj_mix_op, choices["embed_dim"], "proj_embed_dim_msa")
        else:
            self.proj_mixed = nn.ModuleList([])
            for input_dim in choices["embed_dim"]:
                self.proj_mixed.append(MixedLinear(input_dim, input_dim, self.max_embed_dim))
        self.dropout = nn.Dropout(dropout)

    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def forward(self, x, i, arch_params):
        # Simply stack multiple heads
        #out = torch.cat([h(x, i, choices) for h in self.heads], dim=-1)
        head_mixture_out = self.mixop.forward_layer( x, [arch_params["embed_dim"], arch_params["num_heads"][i]], self.heads_mixed, self.heads, combi=True)
        projected_output = self.mixop.forward_layer(head_mixture_out, arch_params["embed_dim"], self.proj_mixed, self.proj)
        return self.dropout(projected_output)