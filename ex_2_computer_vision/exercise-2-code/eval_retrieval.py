import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import paths
from datasets.voc12 import VOCDataset, create_image_only_transforms
from models.blip.blip_config import BlipConfig
from models.blip.blip_retrieval import BlipRetrieval
from models.preprocessing.preprocess import get_processors
from utils.gpu_profiler import GPUProfiler
from utils.logger import setup_logger
from utils.misc import get_timestamp_for_filename
from utils.retrieval_metric import evaluate_retrieval

profiler = GPUProfiler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase verbosity", action="store_true")
    parser.add_argument("--cpu", help="Run on CPU instead of GPU", action="store_true")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=16)
    parser.add_argument("--num_workers", help="Dataloader workers (0 for debugging, higher for "
                                              "faster dataloading)", type=int, default=0)
    parser.add_argument("--output_dir", help="Output directory",
                        type=str, default="outputs/eval_retrieval")
    parser.add_argument("--eval_ckpt", type=str, default=None,
                        help="Path to checkpoint to evaluate")
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
    eval_ckpt = args.eval_ckpt
    if eval_ckpt is None:
        eval_ckpt = Path(paths.CV_PATH_CKPT) / "blip_model_retrieval_head.pth"
    model.load_retrieval_head(eval_ckpt)
    model = model.to(device)
    model.eval()

    logging.info(f"GPU/RAM status: {profiler.profile_to_str()}")

    output_dir = Path(args.output_dir) / get_timestamp_for_filename()
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    # setup dataloader
    voc_path = Path(paths.CV_PATH_VOC)
    logging.info(f"Load dataset from {voc_path}")
    vis_processor_val, text_processor_val = get_processors(model_cfg, mode="eval")
    val_dataset = VOCDataset(
        voc_path, voc_path / "ImageSets" / "Segmentation" / "val.txt", load_captions=True,
        transforms=create_image_only_transforms(vis_processor_val))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False)

    image_feats, text_feats = validate(model, val_dataloader, device)
    val_results = evaluate_retrieval(image_feats, text_feats)
    logging.info(f"Validation results: {val_results}")
    logging.info(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.03f}M")
    with (output_dir / f"scores.json").open("w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)


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
