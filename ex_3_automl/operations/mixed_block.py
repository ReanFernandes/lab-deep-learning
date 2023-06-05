import torch.nn as nn
import torch
from operations.mixed_msa import MixedMultiHeadAttention
from operations.mixed_ff import MixedFeedFoward
from operations.mixed_layer_norm import MixedLayerNorm, MixedLayerNormV2
from optimizers.mixop.entangle import EntangledOp
import itertools

class MixedBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, choices, id, block_size, dropout, mixop, use_we_v2=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.max_embed_dim = max(choices["embed_dim"])
        self.sa = MixedMultiHeadAttention(choices, block_size, dropout, mixop, use_we_v2=use_we_v2)
        self.ffwd = MixedFeedFoward(choices, dropout, mixop, use_we_v2=use_we_v2)
        self.ln1 = torch.nn.LayerNorm(self.max_embed_dim)
        self.ln2 = torch.nn.LayerNorm(self.max_embed_dim)
        if use_we_v2:
            self.ln1_op = MixedLayerNormV2(choices["embed_dim"], self.max_embed_dim, self.ln1)
            self.ln2_op = MixedLayerNormV2(choices["embed_dim"], self.max_embed_dim, self.ln2)
            self.ln1_list = self.get_entangle_ops(self.ln1_op, choices["embed_dim"], "ln1_block_"+str(id))
            self.ln2_list = self.get_entangle_ops(self.ln2_op, choices["embed_dim"], "ln2_block_"+str(id))
        else:
            self.ln1_list = [MixedLayerNorm(e, self.max_embed_dim)
                         for e in choices["embed_dim"]]
            self.ln2_list = [MixedLayerNorm(e, self.max_embed_dim)
                         for e in choices["embed_dim"]]
        self.mixop = mixop


    def get_entangle_ops(self, op, choices, op_name):
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]

    def get_entangle_ops_combi(self, op, choices1, choices2, op_name):
        choices = list(itertools.product(choices1, choices2))
        return [EntangledOp(None, op_name) for _ in range(len(choices)-1)] + [EntangledOp(op, op_name)]
    
    def forward(self, x, i, arch_params=None):
        x = x + self.sa(self.mixop.forward_layer(x, 
                        arch_params["embed_dim"], self.ln1_list, self.ln1), i, arch_params=arch_params)
        x = x + self.ffwd(self.mixop.forward_layer(x, 
                        arch_params["embed_dim"], self.ln2_list,
                        self.ln2), i, arch_params=arch_params)
        return x
