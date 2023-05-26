import torch
from torch.nn import functional as F
import numpy as np


def get_batch(split, train_data, val_data, block_size=128, batch_size=32, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size=128, batch_size=32, eval_iters=100, device="cuda"):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size=block_size, batch_size=batch_size)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_config(n_layers, embed_dims, heads, activation_indices, bias_proj_list, bias_head_list, ffwd_bias_list):
    config = {}
    n_layer = np.random.choice(n_layers)
    config["n_layer"] = n_layer
    n_embd = np.random.choice(embed_dims)
    config["embed_dim"] = n_embd
    n_head = [np.random.choice(heads) for _ in range(n_layer)]
    config["n_head"] = n_head
    activation_id = [np.random.choice(
    activation_indices) for _ in range(n_layer)]
    config["activation_id"] = activation_id
    bias_proj = [np.random.choice(bias_proj_list) for _ in range(n_layer)]
    config["bias_proj"] = bias_proj
    bias_head_q = [np.random.choice(bias_head_list) for _ in range(n_layer)]
    config["bias_head_q"] = bias_head_q
    bias_head_k = [np.random.choice(bias_head_list) for _ in range(n_layer)]
    config["bias_head_k"] = bias_head_k
    bias_head_v = [np.random.choice(bias_head_list) for _ in range(n_layer)]
    config["bias_head_v"] = bias_head_v
    net_0_bias = [np.random.choice(ffwd_bias_list) for _ in range(n_layer)]
    config["net_0_bias"] = net_0_bias
    net_1_bias = [np.random.choice(ffwd_bias_list) for _ in range(n_layer)]
    config["net_1_bias"] = net_1_bias
    lm_head_bias = np.random.choice([True, False])
    config["bias_lm_head"] = lm_head_bias
    head_size = [n_embd//h for h in n_head]
    config["head_size"] = head_size
    return config, n_layer, n_embd, n_head, activation_id, bias_proj, bias_head_q, bias_head_k, bias_head_v, net_0_bias, net_1_bias, lm_head_bias, head_size