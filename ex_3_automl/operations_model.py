import torch
import torch.nn as nn
class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(input)


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, block_size, n_embd, dropout, head_size, bias_proj=False, bias_head=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, block_size, dropout, head_size, bias=bias_head)
                                   for _ in range(num_heads)])  # Slightly more efficient version below
        self.proj = nn.Linear(n_embd, n_embd, bias=bias_proj)
        self.dropout = nn.Dropout(dropout)

    def sample_proj(self, i, choices):
        embed_dim = choices["embed_dim"]
        bias_proj = choices["bias_proj"][i]
        if bias_proj:
            return self.proj.weight[:embed_dim, :embed_dim], self.proj.bias[:embed_dim]
        else:
            return self.proj.weight[:embed_dim, :embed_dim], None

    def forward(self, x, i, choices):
        # Simply stack multiple heads
        out = torch.cat([h(x, i, choices) for h in self.heads], dim=-1)
        weight, bias = self.sample_proj(i, choices)
        out = self.dropout(torch.nn.functional.linear(out[:,:,:choices["embed_dim"]], weight, bias))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout, activation_id=0, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )
        self.activation_id = activation_id
        self.activations = [nn.ReLU(), nn.GELU(), SiLUActivation(), new_gelu]

    def get_weights(self, i, choices):
        bias_net_0 = choices["net_0_bias"][i]
        bias_net_1 = choices["net_1_bias"][i]
        n_embd = choices["embed_dim"]
        if bias_net_0==True and bias_net_1==True:
           return self.net[0].weight[:4*n_embd,:n_embd], self.net[0].bias[:4*n_embd], self.net[1].weight[:n_embd,:4*n_embd], self.net[1].bias[:n_embd]
        elif bias_net_0==True and bias_net_1==False:
              return self.net[0].weight[:4*n_embd,:n_embd], self.net[0].bias[:4*n_embd], self.net[1].weight[:n_embd,:4*n_embd], None
        elif bias_net_0==False and bias_net_1==True:
              return self.net[0].weight[:4*n_embd,:n_embd], None, self.net[1].weight[:n_embd,:4*n_embd], self.net[1].bias[:n_embd]
        else:
              return self.net[0].weight[:4*n_embd,:n_embd], None, self.net[1].weight[:n_embd,:4*n_embd], None

    def forward(self, x, i, choices):
        activation_id = choices["activation_id"][i]
        weight_0, bias_0, weight_1, bias_1 = self.get_weights(i, choices)
        x = torch.nn.functional.linear(x[:,:,:choices["embed_dim"]], weight_0, bias_0)
        x = self.activations[activation_id](x)
        x = torch.nn.functional.linear(x[:,:,:4*choices["embed_dim"]], weight_1, bias_1)
        x = self.net[2](x)
        return x


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, block_size, dropout, head_size, bias=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def get_bias(self,bias_choice, head_size):
        if bias_choice== 'True':
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
        return self.key.weight[:head_size,:embed_dim], self.query.weight[:head_size,:embed_dim], self.value.weight[:head_size,:embed_dim], bias_k, bias_q, bias_v


    def forward(self, x, i, choices):
        B, T, C = x.shape
        kw , qw, vw, bias_k, bias_q, bias_v = self.sample_kqv(i, choices)
        embed_dim = choices['embed_dim']
        k = torch.nn.functional.linear(x[:,:,:embed_dim], kw, bias_k)
        q = torch.nn.functional.linear(x[:,:,:embed_dim], qw, bias_q)
        v  = torch.nn.functional.linear(x[:,:,:embed_dim], vw, bias_v)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, num_heads, block_size, head_size, dropout, bias_proj=False, bias_head=False, ffwd_bias=False, activation_id=0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(
            num_heads, block_size, n_embd, dropout, head_size, bias_proj=bias_proj, bias_head=bias_head)
        self.ffwd = FeedFoward(
            n_embd, dropout, activation_id=activation_id, bias=ffwd_bias)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def get_weights_layernorm(self, choices):
        embed_choice = choices['embed_dim']
        return self.ln1.weight[:embed_choice], self.ln1.bias[:embed_choice], self.ln2.weight[:embed_choice], self.ln2.bias[:embed_choice]

    def forward(self, x, i, choices):
        ln1_weight, ln1_bias, ln2_weight, ln2_bias = self.get_weights_layernorm(choices)
        x = x + self.sa(torch.nn.functional.layer_norm(x,[choices["embed_dim"]],weight=ln1_weight, bias=ln1_bias), i, choices)
        x = x + self.ffwd(torch.nn.functional.layer_norm(x,[choices["embed_dim"]],weight=ln2_weight, bias=ln2_bias), i, choices)
        return x

