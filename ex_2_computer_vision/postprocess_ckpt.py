import argparse
import os
from pathlib import Path

import torch as th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="model_base_capfilt_large.pth")
    parser.add_argument("--output_base", type=str, default="ckpt/blip_model_base.pth")
    parser.add_argument("--output_retrieval_head", type=str,
                        default="ckpt/blip_model_retrieval_head.pth")
    args = parser.parse_args()

    src_file = Path(args.checkpoint)
    tar_file_base = Path(args.output_base)
    tar_file_retrieval_head = Path(args.output_retrieval_head)

    print(f"Loading checkpoint from {src_file}")
    state_dict = th.load(args.checkpoint)["model"]

    print(f"Processing checkpoint")
    new_base, new_retrieval_head = {}, {}
    for key, param in state_dict.items():
        key_start = key.split(".")[0]
        if key_start in ["visual_encoder", "text_encoder", "text_decoder"]:
            new_base[key] = param
        elif key_start in ["vision_proj", "text_proj"]:
            new_retrieval_head[key] = param

    print(f"Saving {len(new_base)} weights to {tar_file_base}")
    os.makedirs(tar_file_base.parent, exist_ok=True)
    th.save(new_base, tar_file_base)

    print(f"Saving {len(new_retrieval_head)} to {tar_file_retrieval_head}")
    os.makedirs(tar_file_retrieval_head.parent, exist_ok=True)
    th.save(new_retrieval_head, tar_file_retrieval_head)

    print("Done")


if __name__ == "__main__":
    main()
