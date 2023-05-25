import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import paths
from datasets import VOCDataset
from datasets.voc12 import create_image_only_transforms
from models.blip.blip_caption import BlipCaption
from models.blip.blip_config import BlipConfig
from models.preprocessing.preprocess import get_processors
from utils.captioning_metric import corpus_bleu
from utils.gpu_profiler import GPUProfiler
from utils.logger import setup_logger
from utils.misc import get_timestamp_for_filename

profiler = GPUProfiler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase verbosity", action="store_true")
    parser.add_argument("--cpu", help="Run on CPU instead of GPU", action="store_true")
    parser.add_argument("--single_image", help="Caption a single image (useful for debugging)",
                        action="store_true")
    parser.add_argument("--use_topk_sampling", help="Sampling instead of greedy decoding",
                        action="store_true")
    parser.add_argument("--topk", help="Value for TopK sampling", type=int, default=50)
    parser.add_argument("--temperature", help="Temperature for sampling", type=float, default=1.0)
    parser.add_argument("--prompt", help="Prompt for captioning", type=str, default="a picture of ")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=16)
    parser.add_argument("--num_workers", help="Dataloader workers (0 for debugging, higher for "
                                              "faster dataloading)", type=int, default=0)
    parser.add_argument("--output_dir", help="Output directory",
                        type=str, default="outputs/eval_captioning")
    args = parser.parse_args()
    setup_logger(level=logging.DEBUG if args.verbose else logging.INFO)
    torch.manual_seed(42)

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Running on device: {device}, cuda available: {torch.cuda.is_available()}")

    if not args.use_topk_sampling:
        if args.topk != 50 or args.temperature != 1.0:
            logging.warning(f"Changing the sampling parameters topk and temperature "
                            f"will have no effect since sampling is disabled.")

    model_cfg = BlipConfig()
    model = BlipCaption.from_config(model_cfg)
    logging.info(f"Created model {type(model).__name__} with {model.show_n_params()} parameters.")
    model.load_checkpoint(Path(paths.CV_PATH_CKPT) / "blip_model_base.pth")
    model = model.to(device)
    model.eval()

    vis_processor, text_processor = get_processors(model_cfg, mode="eval")
    voc_path = Path(paths.CV_PATH_VOC)
    logging.info(f"Load dataset from {voc_path}")
    dataset = VOCDataset(voc_path, voc_path / "ImageSets" / "Segmentation" / "val.txt",
                         load_captions=True, transforms=create_image_only_transforms(vis_processor))

    logging.info(f"GPU/RAM status: {profiler.profile_to_str()}")

    generate_settings = dict(prompt=args.prompt, use_topk_sampling=args.use_topk_sampling,
                             topk=args.topk, temperature=args.temperature)
    if args.single_image:
        data = dataset[0]
        image = data["image"]
        image_in = vis_processor(image)
        images = image_in[None].to(device)  # shape (1, 3, 384, 384)
        logging.info(f"Settings for generation: {generate_settings}")
        with torch.no_grad():
            output = model.generate(images, **generate_settings)
        logging.info(f"Output: {output}")
        return

    output_dir = Path(args.output_dir) / get_timestamp_for_filename()
    logging.info(f"Output dir: {output_dir}")

    # caption the whole dataset and compare with the given captions
    ref_captions, pred_captions = [], []
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    pbar = tqdm(total=len(dataloader), desc="Captioning images")
    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        captions = batch["caption"]
        ref_captions.extend(captions)
        generate_settings = dict(prompt=args.prompt, use_topk_sampling=args.use_topk_sampling,
                                 topk=args.topk, temperature=args.temperature)
        with torch.no_grad():
            output = model.generate(images, **generate_settings)
        if i == 0:
            with logging_redirect_tqdm():
                logging.info(f"GPU/RAM status: {profiler.profile_to_str()}")
                logging.info(f"Datapoint {i} got output {output}")
        pred_captions.extend(output)
        pbar.update()
    pbar.close()

    os.makedirs(output_dir, exist_ok=True)
    pred_captions_file = output_dir / "pred_captions.txt"
    write_captions(pred_captions, pred_captions_file)
    ref_captions_file = output_dir / "ref_captions.txt"
    write_captions(ref_captions, ref_captions_file)

    logging.info("Evaluating captions")
    bleu_score = evaluate_captions(ref_captions_file, pred_captions_file)
    score_dict = {"bleu": bleu_score}
    logging.info(f"Scores: {score_dict}")
    score_file = output_dir / "scores.json"
    logging.info(f"Writing scores to {score_file}")
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, indent=4)


def evaluate_captions(ref_file, pred_file):
    refs = Path(ref_file).read_text(encoding="utf-8").splitlines()
    preds = Path(pred_file).read_text(encoding="utf-8").splitlines()
    assert len(refs) == len(preds), (
        f"Number of references ({len(refs)}) and predictions ({len(preds)}) do not match.")

    # note ref_list will be a list of list (possibly multiple references can exist)
    # and pred_list will be a list
    refs_list = []
    pred_list = []
    for ref, pred in zip(refs, preds):
        refs_list.append([ref.strip().lower()])
        pred_list.append(pred.strip().lower())
    score = corpus_bleu(refs_list, pred_list)
    return score


def write_captions(captions, file):
    logging.info(f"Writing {len(captions)} captions to {file}")
    with open(file, "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(f"{caption}\n")


if __name__ == "__main__":
    main()
