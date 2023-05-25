import math
import torch
from ..vit.vit import VisionTransformer
from functools import partial


# Implementation of the Vision Transformer (ViT) as an encoder for segmentation.
# It largely takes from the standard image transformer, with some differences:
# - the patch embedding module can handle arbitrary input sizes and aspect ratios
# - the positional embeddings can be resampled to arbitrary input sizes and aspect ratios
# - the output doesn't contain the class token, and is reshaped to the usual feature shape for CNNs: (batch_size, channels, height, width)
# - all parameters can be easily frozen using freeze()

class PatchEmbed(torch.nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTSegmEncoder(VisionTransformer):
    def __init__(self, *args, img_size=224, patch_size=16, in_chans=3, embed_dim=768, frozen=False, **kwargs):
        super().__init__(*args, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, **kwargs)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_heads = self.blocks[-1].attn.num_heads

        if frozen:
            self.freeze()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        return x

    def forward(self, x):
        B, _, H, W = x.shape
        HP, WP = H // self.patch_embed.patch_size, W // self.patch_embed.patch_size
        x = self.prepare_tokens(x)
        C = x.shape[-1]
        # we return the output tokens from the last block
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                # exclude cls token, reshape to conventional feature map shape
                output = self.norm(x)[:, 1:].permute(0, 2, 1).view(B, C, HP, WP)
        return output
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.train(False)


# constructor for an instance of ViT-small with frozen parameters
def vit_small():
    return ViTSegmEncoder(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, frozen=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
