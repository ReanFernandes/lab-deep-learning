import json
import os
import os.path as osp
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# Dataset class for Pascal VOC

class VOCDataset(Dataset):
    classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root, list_path, transforms=None, ignore_label=255, load_captions=False):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.transforms = transforms
        self.img_ids = [i_id.strip() for i_id in Path(list_path).open(encoding="utf-8")]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = (Path(self.root) / "JPEGImages" / f"{name}.jpg").as_posix()
            label_file = (Path(self.root) / "SegmentationClass" / f"{name}.png").as_posix()
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.load_captions = load_captions

        if self.load_captions:
            caption_file = Path(self.root) / f"Captions" / "captions_v1.json"
            caption_data = json.load(caption_file.open(encoding="utf-8"))
            for file in self.files:
                name = file["name"]
                file["caption"] = caption_data[name]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        output = {
            "image": image,
            "label": label,
            "name": name,
            "img_path": datafiles["img"]
        }
        if self.load_captions:
            output["caption"] = datafiles["caption"]
        return output


def create_image_only_transforms(image_transforms):
    def transform_image_only(image, _label):
        """
        in case the segmentation label is not used,
        transform the image and return a string for the label
        """
        image_out = image_transforms(image)
        return image_out, "no_label"
    return transform_image_only
