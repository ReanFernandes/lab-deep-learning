import collections
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import argparse

import torch

import contextlib
import pickle
from model_search import BigramLanguageModel
# Encoder: take a string, output a list of integers


def encode(s):
    return [stoi[c] for c in s]

# Decoder: take a list of integers, output a string


def decode(l):
    return ''.join([itos[i] for i in l])


global data, train_data, valid_data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Checking all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab_set = "".join(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Train and test splits
train_size = 0.9
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_size * len(data))
train_data = data[:n]
valid_data = data[n:]
train_portion = 0.8
n_train = int(train_portion * len(train_data))
#print(n_train)
#print(len(train_data))
train_data_now = train_data[:n_train]
#print(len(train_data))
eval_data = train_data[n_train:]
#print(len(eval_data))



def get_batch(split: str, block_size: int = 8, batch_size: int = 4, device: str = None):
    """ Gets a randomized batch from the split of data chosen.

    Arguments
    ---------
    split : str, {"train", "valid"}
    block_size : int
        The context length for predictions, that is, a sentence length
    batch_size : int
        The batch size, that is, the number of sentences
    """
    # generate a small batch of data of inputs x and targets y
    assert split in ["train", "valid", "test"]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = train_data if split == 'train' else eval_data
    if split == "test":
        data = valid_data
    # generating random indices as markers in the full text document
    # such that they are a starting point to the sentence of length
    # `block_size` that will be a data point in the batch
    ix = torch.randint(
        low=0, high=len(data) - block_size, size=(batch_size,)
    )
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix`
    x = torch.stack([data[i:i+block_size] for i in ix])
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix` + 1 (shifted to right)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class NASOptimizer(object):
    """
    Base class for NASBench-101 optimizers. All subclasses should
    inherit from this.
    """

    def __init__(self):
        # get the configuration space
        # configuration (architecture) at each point in time.
        # incumbent_trajectory_error keeps track of the
        # corresponding validation errors of incumbent_trajectory
        self.incumbent_trajectory = []
        self.incumbent_trajectory_error = []
        self.incumbent_trajectory_test_error = []
        self.all_configs_err = {}
        self.curr_wallclock = 0
        self.curr_incumbent = None
        self.curr_incumbent_error = 10000000
        self.eval_iters = 200

    def optimize(self, n_iters: int = 100):
        raise NotImplementedError

    def sample_random_config(self, model):
        """
        Return a randomly sampled configuration.
        """
        # TODO: return one randomly sampled configuration from self.cs
        config, arch_params = model.sample_random_config()
        return config, arch_params

    @torch.no_grad()
    def estimate_loss(self, arch_params, model):
        out = {}
        model.eval()
        for split in ['valid','test']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y, arch_params=arch_params)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out['valid'], out['test']

    def train_and_eval(self, config, arch_params, model):
        """
        Function that computes the error on the validation split. 
        Since every architecture has already been trained partially, 
        we just to forward props on the pre-trained supernet 
        """
        valid_err, test_err = self.estimate_loss(arch_params, model)
        # TODO: check if config is better than current incumbent
        self.all_configs_err[str(config)] = valid_err
        # If we find a better validation error, we update the incumbent, else revet to the best current incumbent
        if min(self.curr_incumbent_error, valid_err) == valid_err:
            self.curr_incumbent_error = valid_err
            self.curr_incumbent_test_error = test_err
            self.curr_incumbent = config
            self.incumbent_trajectory.append(config)
            self.incumbent_trajectory_error.append(valid_err)
            self.incumbent_trajectory_test_error.append(test_err)
        else:
            self.incumbent_trajectory.append(self.curr_incumbent)
            self.incumbent_trajectory_error.append(
                self.incumbent_trajectory_error[-1])
            self.incumbent_trajectory_test_error.append(
                self.incumbent_trajectory_test_error[-1])
        print("Current incumbent error: ", self.curr_incumbent_error)
        print("Current incumbent test error: ", self.curr_incumbent_test_error)
        print("Current incumbent: ", self.curr_incumbent)
        with open("incumbent_trajectory_error_rs.pkl", "wb") as f:
            pickle.dump(self.incumbent_trajectory_error, f)
        with open("incumbent_trajectory_rs.pkl", "wb") as f:
            pickle.dump(self.incumbent_trajectory, f)
        with open("incumbent_trajectory_test_error_rs.pkl", "wb") as f:
            pickle.dump(self.incumbent_trajectory_test_error, f)
        with open("all_configs_err_rs.pkl", "wb") as f:
            pickle.dump(self.all_configs_err, f)


class RandomSearch(NASOptimizer):
    """
    Algorithm for random search.
    """

    def __init__(self, model_path):
        super(RandomSearch, self).__init__()
        self.model_path = model_path
        choices = {}

        choices["num_layers"] = [2, 4, 6]
        choices["embed_dim"] = [96, 192, 384]
        choices["num_heads"] = [2, 4, 6]
        choices["mlp_ratio"] = [1, 2, 4]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BigramLanguageModel(choices=choices, block_size=256,
                                    mixop="spos", dropout=0.2, use_we_v2=False).to(device)
        self.model.load_state_dict(torch.load(model_path))

    def optimize(self, n_iters: int = 100):
        """
        Run random search for n_iters function evaluations.
        """
        for i in range(n_iters):
            config, arch_params = self.sample_random_config(self.model)
            self.train_and_eval(config, arch_params, self.model)


rs = RandomSearch("model_one_shot_spos_pretrained.pth")
rs.optimize(100)
