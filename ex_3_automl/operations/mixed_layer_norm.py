import torch
import torch.nn as nn
class MixedLayerNorm(nn.Module):
    def __init__(self, embed_dim, max_embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, layer_norm):
        weight = layer_norm.weight[:self.embed_dim]
        bias = layer_norm.bias[:self.embed_dim]
        return weight, bias
    
    def forward(self, x, layer_norm):
        weight, bias = self.sample_weights_and_bias(layer_norm)
        out =  torch.nn.functional.layer_norm(x[:,:,:self.embed_dim], [self.embed_dim], weight=weight, bias=bias)
        # pad output
        if out.shape[-1] != self.max_embed_dim:
            out = torch.nn.functional.pad(out, (0,self.max_embed_dim - out.shape[-1]), "constant", 0)
        return out

class MixedLayerNormV2(nn.Module):
    def __init__(self, embed_dim_list, max_embed_dim, layer_norm) -> None:
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.max_embed_dim = max_embed_dim
        self.layer_norm = layer_norm

    def sample_weights_and_bias(self, emb_dim):
        weight = self.layer_norm.weight[:emb_dim]
        bias = self.layer_norm.bias[:emb_dim]
        return weight, bias
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            weight, bias = self.sample_weights_and_bias(self.embed_dim_list[weights_max])
            # pad weights and bias
            weight = weights[weights_max]*torch.nn.functional.pad(weight, (0,self.max_embed_dim - weight.shape[-1]), "constant", 0)
            bias =  weights[weights_max]*torch.nn.functional.pad(bias, (0,self.max_embed_dim - bias.shape[-1]), "constant", 0)
            out =  torch.nn.functional.layer_norm(x, [self.max_embed_dim], weight=weight, bias=bias)
            return out
        else:
            weights_mix = 0
            bias_mix = 0
            for i, embed_dim in enumerate(self.embed_dim_list):
                weight, bias = self.sample_weights_and_bias(embed_dim)
                # pad weights and bias
                weight = torch.nn.functional.pad(weight, (0,self.max_embed_dim - weight.shape[-1]), "constant", 0)
                bias = torch.nn.functional.pad(bias, (0,self.max_embed_dim - bias.shape[-1]), "constant", 0)
                weights_mix += weights[i]*weight
                bias_mix += weights[i]*bias
            out =  torch.nn.functional.layer_norm(x, [self.max_embed_dim], weight=weights_mix, bias=bias_mix)
            return out
    
'''# test mixed layer norm
layer_norm = nn.LayerNorm(10)
input = torch.randn(2, 2, 10)
mixed_layer_norm = MixedLayerNorm(5, 10)
out = mixed_layer_norm(input, layer_norm)
print(out.shape)'''
