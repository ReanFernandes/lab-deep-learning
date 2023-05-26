import torch
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
from model_search import BigramLanguageModel
from utils import *
import argparse
# main function
# -------------
# load the dataset
if __name__ == "__main__":
    #check if output directory is created
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create summary writer
    writer = SummaryWriter(log_dir=output_dir)

    parser = argparse.ArgumentParser(
        'CharLM training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--block_size', default=256, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--eval-interval', default=10, type=int)
    parser.add_argument('--eval-iters', default=200, type=int)
    parser.add_argument('--mixop', default='spos', type=str)
    parser.add_argument('--train_iters', default=50000, type=int)
    parser.add_argument('--portion', default=0.5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print("length of dataset in characters: ", len(text))
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    # encoder: take a string, output a list of integers
    def encode(s): return [stoi[c] for c in s]
    # decoder: take a list of integers, output a string
    def decode(l): return ''.join([itos[i] for i in l])
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    train_portion = args.portion
    n_train = int(train_portion * len(train_data))
    train_data = train_data[:n_train]
    # how many independent sequences will we process in parallel?
    batch_size = args.batch_size
    block_size = args.block_size  # what is the maximum context length for predictions?
    eval_interval = args.eval_interval
    learning_rate = args.lr
    eval_iters = args.eval_iters
    dropout = args.dropout
    mixop = args.mixop
    # ------------
    choices = {}
    choices["num_layers"] = [2, 4, 6]
    choices["embed_dim"] = [96, 192, 384]
    choices["num_heads"] = [2, 4, 6]
    choices["mlp_ratio"] = [1, 2, 4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BigramLanguageModel(
        choices=choices, block_size=block_size, mixop=mixop).to(device)
    # model = torch.nn.DataParallel(model).to(device)
    optimizer_model = torch.optim.AdamW(
        model.get_model_parameters(), lr=learning_rate)
    #optimizer_arch = torch.optim.Adam(model.get_arch_parameters(), betas=(0.5, 0.999), lr=learning_rate, weight_decay=1e-3)
    model_save_path = 'model_one_shot_{}_{}_{}.pth'.format(mixop, str(train_portion), str(args.seed))
    torch.manual_seed(args.seed)
    model.sampler.set_taus(0.1, 10)
    model.sampler.set_total_epochs(args.train_iters//100)
    alphas = model.sampler.sample_epoch(model.get_arch_parameters())
    print("Args: ", args)
    for i in range(args.train_iters):
        xb, yb = get_batch('train', train_data, val_data,
                             block_size=block_size, batch_size=batch_size, device=device)
        # evaluate the loss
        model.to(device)
        xb = xb.to(device)
        yb = yb.to(device)
        # optimizer_model.step()
        optimizer_model.zero_grad()
        logits, loss = model(xb, targets=yb, arch_params=alphas)
        loss.backward()
        optimizer_model.step()
        optimizer_model.zero_grad()
        if i % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, block_size=block_size,
                                   batch_size=batch_size, eval_iters=eval_iters, device=device)
            print(
                f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            writer.add_scalar('train_loss', losses['train'], i)
            writer.add_scalar('val_loss', losses['val'], i)
            arch_params = list(model.arch_parameter_dict.values())
            for param in arch_params:
                print(torch.nn.functional.softmax(param, dim=-1))
            torch.save(model.state_dict(), model_save_path)
        alphas = model.sampler.sample_epoch(model.get_arch_parameters())