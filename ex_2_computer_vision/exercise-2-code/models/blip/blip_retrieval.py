import logging

from torch import nn
import torch
import torch.nn.functional as F

from models.bert.bert import XBertEncoder
from models.blip.blip import BlipBase
from models.blip.blip_config import BlipConfig
from models.vit.vit import VisionTransformerEncoder


class BlipRetrieval(BlipBase):
    def __init__(
            self,
            image_encoder,
            text_encoder,
            embed_dim=128,
            max_txt_len=35,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # projection layers for image-text contrastive learning
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.max_txt_len = max_txt_len

    @classmethod
    def from_config(cls, cfg: BlipConfig):
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)
        embed_dim = cfg.embed_dim
        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
        )
        return model

    def load_retrieval_head(self, filename):
        state_dict = torch.load(filename, map_location="cpu")
        msg = self.load_state_dict(state_dict, strict=False)
        for key in msg.missing_keys:
            assert not key.startswith("vision_proj") and not key.startswith("text_proj")
        logging.info(f"Done loading retrieval head from {filename}")

    def get_retrieval_head_state_dict(self):
        retrieval_state_dict = {}
        for key, param in self.state_dict().items():
            key_start = key.split(".")[0]
            if key_start in ("vision_proj", "text_proj"):
                retrieval_state_dict[key] = param
        return retrieval_state_dict

    def forward(self, image, caption):
        """
        Args:
            image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images.
            caption (list): A list of length batch_size, each element is a string of text/caption.

        Returns:
            image_feat (torch.Tensor): A tensor of shape (batch_size, embed_dim). The image features.
            text_feat (torch.Tensor): A tensor of shape (batch_size, embed_dim). The text features.
        """
        image_feat = self.forward_image(image)
        text_feat = self.forward_text(caption)

        return image_feat, text_feat

    def forward_image(self, image):
        # START TODO #################
        # 1. Pass the image through the visual encoder to get the image embeddings
        # 2. Select the CLS token output (the first token in the sequence)
        # 3. Run it through the visual projection layer
        # 4. Normalize it with F.normalize
        image_embeds = self.visual_encoder.forward_features(image)
        cls_token = image_embeds[:, 0]
        image_feat = self.vision_proj(cls_token)
        image_feat = F.normalize(image_feat, dim=-1)
        # END TODO ###################

        return image_feat

    def forward_text(self, caption):
        # tokenize text and forward it through the text encoder
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text_embeds = self.text_encoder.forward_text(text)

        # START TODO #################
        # 1. Select the CLS token output (the first token in the sequence)
        # 2. Run it through the text projection layer
        # 3. Normalize it with F.normalize
        cls_token = text_embeds[:, 0]
        text_feat = self.text_proj(cls_token)
        text_feat = F.normalize(text_feat, dim=-1)
        # END TODO ###################

        return text_feat
