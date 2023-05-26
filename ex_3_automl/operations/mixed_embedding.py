import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MixedEmbedding(nn.Module):
    def __init__(self, embed_dim, max_embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, embedding):
        weight = embedding.weight[:,:self.embed_dim]
        return weight
    
    def forward(self, x, embedding):
        weight = self.sample_weights_and_bias(embedding)
        out = F.embedding(x, weight)
        # pad output
        if out.shape[-1] != self.max_embed_dim:
            out = F.pad(out, (0,self.max_embed_dim - out.shape[-1]), "constant", 0)
        return out
    
class MixedEmbeddingV2(nn.Module):
    def __init__(self, embed_dim_list, max_embed_dim, embedding) -> None:
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.embedding = embedding
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, embed_dim):
        weight = self.embedding.weight[:,:embed_dim]
        return weight
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            emb_weight = self.sample_weights_and_bias(self.embed_dim_list[weights_max])
            emb_weight_padded = F.pad(emb_weight, (0,self.max_embed_dim - emb_weight.shape[-1]), "constant", 0)
            out = weights[weights_max]*F.embedding(x, emb_weight_padded)
            return out
        else:
            embedding_mixture = 0
            i=0
            for embed_dim in self.embed_dim_list:
                emb_weight = self.sample_weights_and_bias(embed_dim)
                emb_weight_padded = F.pad(emb_weight, (0,self.max_embed_dim - emb_weight.shape[-1]), "constant", 0)
                embedding_mixture += weights[i] * emb_weight_padded
                i+=1
            out = F.embedding(x, embedding_mixture)
            return out

# test mixed embedding
'''embedding = nn.Embedding(10, 20)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
mixed_embedding = MixedEmbedding(5, 20)
out = mixed_embedding(input, embedding)
print(out.shape)'''