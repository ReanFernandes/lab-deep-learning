"""
Based on salesforce blip config YAML file.

 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from dataclasses import dataclass, field
from typing import Tuple

from models.bert.bert_config import BertConfig


@dataclass
class BlipConfig:
    # vision encoder
    vit_type: str = "base"
    image_size: int = 384
    vit_drop_path_rate: float = 0.0
    vit_norm_layer_eps: int = -1

    # text model
    bert_cfg: BertConfig = field(default_factory=BertConfig)

    # text decoder settings
    prompt: str = "a picture of"
    max_txt_len: int = 40

    # retrieval settings
    embed_dim: int = 256

    # dataloader settings
    preprocess_vis_mean: Tuple[float, float, float] = None
    preprocess_vis_std: Tuple[float, float, float] = None
    preprocess_text_max_words: int = 50
    preprocess_vis_min_scale: float = 0.5
    preprocess_vis_max_scale: float = 1.0
