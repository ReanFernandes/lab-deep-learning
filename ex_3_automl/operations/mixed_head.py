import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, block_size, dropout, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def get_bias(self, bias_choice, head_size):
        if bias_choice == 'True':
            bias = self.key.bias[:head_size]
        else:
            bias = None
        return bias

    def sample_kqv(self, i, choices):
        head_size = choices['head_size'][i]
        embed_dim = choices['embed_dim']
        bias_k = self.get_bias(choices['bias_head_k'][i], head_size)
        bias_q = self.get_bias(choices['bias_head_q'][i], head_size)
        bias_v = self.get_bias(choices['bias_head_v'][i], head_size)
        return self.key.weight[:head_size, :embed_dim], self.query.weight[:head_size, :embed_dim], self.value.weight[:head_size, :embed_dim], bias_k, bias_q, bias_v

    def forward(self, x, i, choices):
        B, T, C = x.shape
        kw, qw, vw, bias_k, bias_q, bias_v = self.sample_kqv(i, choices)
        embed_dim = choices['embed_dim']
        k = torch.nn.functional.linear(x[:, :, :embed_dim], kw, bias_k)
        q = torch.nn.functional.linear(x[:, :, :embed_dim], qw, bias_q)
        v = torch.nn.functional.linear(x[:, :, :embed_dim], vw, bias_v)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MixedHeads(nn.Module):
    def __init__(self, embed_dim, num_heads, max_embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.max_embed_dim = max_embed_dim

    def sample_kqv(self, base_head):
        return base_head.key.weight[:self.head_size, :self.embed_dim], base_head.query.weight[:self.head_size, :self.embed_dim], base_head.value.weight[:self.head_size, :self.embed_dim]

    def forward(self, x, base_heads):
        heads_considered = base_heads[:self.num_heads]
        B, T, C = x.shape
        head_out_list = []
        for head in heads_considered:
            weights_k, weights_q, weights_v = self.sample_kqv(head)
            k = torch.nn.functional.linear(
                x[:, :, :self.embed_dim], weights_k, bias=None)
            q = torch.nn.functional.linear(
                x[:, :, :self.embed_dim], weights_q, bias=None)
            v = torch.nn.functional.linear(
                x[:, :, :self.embed_dim], weights_v, bias=None)
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(
                head.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = head.dropout(wei)
            # perform the weighted aggregation of the values
            out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
            head_out_list.append(out)
        out_stacked = torch.cat(head_out_list, dim=-1)
        out_stacked = torch.nn.functional.pad(
            out_stacked, (0, self.max_embed_dim - out_stacked.shape[-1]))
        return out_stacked


# FIXME : The version below gives an error for now default back to we2 heads mixture

class MixedHeadsV2(nn.Module):
    def __init__(self, embed_dim_list, num_heads_list, max_embed_dim, base_heads):
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.num_heads_list = num_heads_list
        self.max_head_size = max(embed_dim_list) // min(num_heads_list)
        self.max_embed_dim = max_embed_dim
        self.base_heads = base_heads

    def sample_kqv(self, base_head, head_size, embed_dim):
        return base_head.key.weight[:head_size, :embed_dim], base_head.query.weight[:head_size, :embed_dim], base_head.value.weight[:head_size, :embed_dim]

    def forward(self, x, weights, use_argmax=False):
        weights_k_dict = {}
        weights_q_dict = {}
        weights_v_dict = {}
        for i, head in enumerate(self.base_heads):
            weights_k_dict[i], weights_q_dict[i], weights_v_dict[i] = 0, 0, 0
        i = 0
        for emb in self.embed_dim_list:
            for heads in self.num_heads_list:
                heads_considered = self.base_heads[:heads]
                head_size = emb // heads
                #print("Head size: ", head_size)
                #print("Num heads: ", heads)
                #print("Embed dim: ", emb)
                B, T, C = x.shape
                for h, head in enumerate(heads_considered):
                    weights_k, weights_q, weights_v = self.sample_kqv(
                        head, head_size, emb)
                    # Pad weights
                    weights_k_padded = torch.nn.functional.pad(
                        weights_k, ( 0, self.max_embed_dim-weights_k.shape[-1], 0, self.max_head_size - weights_k.shape[-2]), mode="constant", value=0)
                    weights_q_padded = torch.nn.functional.pad(
                        weights_q, ( 0, self.max_embed_dim-weights_k.shape[-1], 0, self.max_head_size - weights_k.shape[-2]), mode="constant", value=0)
                    weights_v_padded = torch.nn.functional.pad(
                        weights_v, (0, self.max_embed_dim-weights_k.shape[-1], 0, self.max_head_size - weights_k.shape[-2]), mode="constant", value=0)
                    #print(weights_k_padded.shape, weights_q_padded.shape, weights_v_padded.shape)
                    #print(weights_k.shape, weights_q.shape, weights_v.shape)
                    weights_k_dict[h] += weights[i]*weights_k_padded
                    weights_q_dict[h] += weights[i]*weights_q_padded
                    weights_v_dict[h] += weights[i]*weights_v_padded
                i += 1
        head_out_list = []
        for i, head in enumerate(self.base_heads):
            weights_k = weights_k_dict[i]
            weights_q = weights_q_dict[i]
            weights_v = weights_v_dict[i]
            k = torch.nn.functional.linear(x, weights_k, bias=None)
            q = torch.nn.functional.linear(x, weights_q, bias=None)
            v = torch.nn.functional.linear(x, weights_v, bias=None)
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(
                head.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = head.dropout(wei)
            # perform the weighted aggregation of the values
            out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
            head_out_list.append(out)
        out_stacked = torch.cat(head_out_list, dim=-1)
        return out_stacked


'''# Initialize head base
head_base = Head(n_embd=64, block_size=10, dropout=0.1, head_size=32, bias=False)
print(head_base)
# Initialize mixed head
mixed_head = MixedHead(n_embd=32, num_heads=2, max_head_dim=32)
# Initialize dummy 
idx = torch.randn(1, 10, 64)
print(idx.shape)
# Forward pass
out = mixed_head(idx, head_base)
print(out.shape)'''
