import torch
import torch.nn as nn
from ..vit.vit import Mlp, Attention


# Attention mechanism that uses the same projection for queries and keys.

class AttentionSameQK(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = args[0]
        qkv_bias=False
        self.qkv = None
        # START TODO #################
        # re-build the self.qkv layer for shared Q and K projections
        self.qkv = nn.Linear(in_features=dim, out_features=dim*2, bias=qkv_bias)
        # END TODO ###################
    
    def forward(self, x):
        # START TODO #################
        # re-implement the forward pass with shared Q and K. See models/vit/vit.py
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[0], qkv[1] # shared Q and K
        # END TODO ###################

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Implementation of a transformer head for segmentation, using extra learnable class embeddings.

class TransformerSegmentationHead(nn.Module):
    def __init__(self, num_classes, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_qk=False):
        super().__init__()
        self.num_classes = num_classes

        # START TODO #################
        # add learnable class parameters/embeddings
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, dim))

        # END TODO ###################

        # normalization layer before attention
        self.norm1 = norm_layer(dim)

        # build attention module
        attention_mechanism = Attention if not shared_qk else AttentionSameQK
        self.attn = attention_mechanism(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        # normalization layer after attention
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        # build TWO MLPs: one for the patch tokens, one for the class tokens
        self.patches_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.classes_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        # final normalization layer
        self.mask_norm = norm_layer(num_classes)

    def forward(self, x):
        # reshape (patch) embeddings from encoder
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)

        # START TODO #################
        # concatenate patch embeddings with class embeddings
        class_tokens = self.class_embeddings.repeat(b, 1, 1)
        x = torch.cat([x, class_tokens], dim=1)

        # END TODO ###################

        # START TODO #################
        # normalize and compute self attention
        f_x = self.norm1(x)
        f_x = self.attn(f_x)
       
        # END TODO ###################

        # START TODO #################
        # residual connection and normalization
        x = x + f_x 
        x = self.norm2(x)
        

        
        # END TODO ###################

        # START TODO #################
        # split patch and class embeddings, apply mlps
        patch_embeds, class_embeds = torch.split(x, [h*w, self.num_classes],dim=1)
        patch_tokens = self.patches_mlp(patch_embeds)
        class_tokens = self.classes_mlp(class_embeds)
        # END TODO ###################

        # START TODO #################
        # compute segmentation masks via patch-class similarity (you can use matmul or the @ operator) and normalize. 
        masks = patch_tokens @ class_tokens.permute(0, 2, 1)
        masks = self.mask_norm(masks)
        # END TODO ###################

        # reshape masks from (B, H*W, C) to (B, C, H, W)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return masks
