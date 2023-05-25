import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import paths
from datasets.voc12 import VOCDataset, create_image_only_transforms
from eval_retrieval import evaluate_retrieval
from models.blip.blip_config import BlipConfig
from models.blip.blip_retrieval import BlipRetrieval
from models.preprocessing.preprocess import get_processors
from utils.gpu_profiler import GPUProfiler
from utils.logger import setup_logger
from utils.misc import get_timestamp_for_filename

profiler = GPUProfiler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase verbosity", action="store_true")
    parser.add_argument("--cpu", help="Run on CPU instead of GPU", action="store_true")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=16)
    parser.add_argument("--num_workers", help="Dataloader workers (0 for debugging, higher for "
                                              "faster dataloading)", type=int, default=0)
    parser.add_argument("--output_dir", help="Output directory",
                        type=str, default="outputs/train_retrieval")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--finetune", help="Finetune the retrieval head instead of training it "
                                           "from scratch.", action="store_true")
    args = parser.parse_args()
    setup_logger(level=logging.DEBUG if args.verbose else logging.INFO)
    torch.manual_seed(42)

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Running on device: {device}, cuda available: {torch.cuda.is_available()}")

    model_cfg = BlipConfig()
    model = BlipRetrieval.from_config(model_cfg)
    logging.info(f"Created model {type(model).__name__} with {model.show_n_params()} parameters.")
    model.load_checkpoint(Path(paths.CV_PATH_CKPT) / "blip_model_base.pth")
    if args.finetune:
        model.load_retrieval_head(Path(paths.CV_PATH_CKPT) / "blip_model_retrieval_head.pth")
    model = model.to(device)
    model.train()

    # freeze all weights except the projection heads
    trainable_params = []
    for key, param in model.named_parameters():
        key_start = key.split(".")[0]
        if key_start not in ["vision_proj", "text_proj"]:
            param.requires_grad = False
        else:
            trainable_params.append(param)
            logging.info(f"Will train {key} with shape {param.shape}")

    # setup dataloader
    voc_path = Path(paths.CV_PATH_VOC)
    logging.info(f"Load dataset from {voc_path}")
    vis_processor_train, text_processor_train = get_processors(model_cfg, mode="train")
    train_dataset = VOCDataset(
        voc_path, voc_path / "ImageSets" / "Segmentation" / "train.txt", load_captions=True,
        transforms=create_image_only_transforms(vis_processor_train))
    vis_processor_val, text_processor_val = get_processors(model_cfg, mode="eval")
    val_dataset = VOCDataset(
        voc_path, voc_path / "ImageSets" / "Segmentation" / "val.txt", load_captions=True,
        transforms=create_image_only_transforms(vis_processor_val))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False)

    logging.info(f"GPU/RAM status: {profiler.profile_to_str()}")

    output_dir = Path(args.output_dir) / get_timestamp_for_filename()
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")
    tb_logger = SummaryWriter(log_dir=output_dir)

    temperature = args.temperature
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=args.epochs * len(train_dataloader), power=1.0)
    loss_fn = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(args.epochs):
        logging.info(f"Training epoch {epoch}")
        train_epoch(epoch, model, train_dataloader, optimizer, scheduler, loss_fn, device,
                    temperature, tb_logger)
        image_feats, text_feats = validate(model, val_dataloader, device)
        val_results = evaluate_retrieval(image_feats, text_feats)
        logging.info(f"Validation results: {val_results}")
        for k, v in val_results.items():
            tb_logger.add_scalar(f"val/{k}", v, (epoch + 1) * len(train_dataloader))
        torch.save(model.get_retrieval_head_state_dict(), output_dir / f"model_e{epoch + 1}.pth")
        with (output_dir / f"scores_e{epoch + 1}.json").open("w", encoding="utf-8") as f:
            json.dump(val_results, f, indent=2)


def train_epoch(epoch, model, dataloader, optimizer, scheduler, loss_fn, device, temperature,
                tb_logger):
    epoch_step = epoch * len(dataloader)
    tb_logger.add_scalar("train/epoch", epoch, epoch_step)
    print_interval = 10
    gpu_print_interval = 50
    model.train()
    pbar = tqdm(total=len(dataloader), desc=f"Training")
    for step, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        caption = batch["caption"]

        # forward pass
        image_feat, text_feat = model(image, caption)

        # START TODO #################
        # 1. Compute image-to-text and text-to-image similarity matrices (without softmax).
        # Don't forget to divide them by the temperature parameter.
        # 2. Define the one-hot targets, an identity matrix (take care to select the correct device
        # and dtype)
        # 3. Compute the image-to-text and text-to-image cross-entropy loss.
        # 4. Compute the final loss as weighted average of the two losses.
        

        img_to_text_sim = image_feat @ text_feat.T / temperature
        text_to_img_sim = text_feat @ image_feat.T / temperature
        one_hot_targets_i2t = torch.eye(img_to_text_sim.shape[0], device=device, dtype=torch.float32)
        one_hot_targets_t2i = torch.eye(text_to_img_sim.shape[0], device=device, dtype=torch.float32)
        loss_i2t = loss_fn(img_to_text_sim, one_hot_targets_i2t)
        loss_t2i = loss_fn(text_to_img_sim, one_hot_targets_t2i)
        
        loss = (loss_i2t + loss_t2i) / 2
        # END TODO ###################

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_interval == 0:
            global_step = epoch_step + step
            loss_value = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            with logging_redirect_tqdm():
                logging.info(f"  step: {step} loss: {loss_value:.3f} lr: {lr:.3e}")
            tb_logger.add_scalar("train/loss", loss_value, global_step)
            tb_logger.add_scalar("train/lr", lr, global_step)
        if step % gpu_print_interval == 0:
            logging.info(f"GPU/RAM status: {profiler.profile_to_str()}")

        scheduler.step()
        pbar.update(1)
    logging.info(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.03f}M")


def validate(model, dataloader, device):
    model.eval()
    pbar = tqdm(total=len(dataloader), desc=f"Generating retrieval features for eval")

    image_feats, text_feats = [], []
    for i, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        caption = batch["caption"]
        with torch.no_grad():
            image_feat, text_feat = model(image, caption)
            text_feats.append(text_feat.detach().cpu().numpy())
            image_feats.append(image_feat.detach().cpu().numpy())
        pbar.update(1)
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)
    return image_feats, text_feats


if __name__ == "__main__":
    main()
