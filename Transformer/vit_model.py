# vit_model.py
"""
Reference: https://huggingface.co/docs/transformers/main/en/tasks_explained
TODO: Handling the feature patch: resize the patch to the same size. [PatchEmbedding]
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification # This one is for Transformer comparison
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import *
from kornia.core.check import KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE
from SIFT.get_patch import get_resized_patch
from SIFT.scale_angle_rotation import get_laf_scale_and_angle

from embedding import RelativeAttentionBias


"""
Patch Embedding: this step mimic the idea of NLP Transformer to handle the images. 
1. projection: using a convolution layer to convert it into a 14x14 patches. 
2. flattern: flattern the 2d patches into a single column
3. Transpose: modify the dimension to meet the requirement from the latter steps. 

Problem: there is no in_channels=3 anymore. 
"""

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2 # //: round down the division result
        # project the input into higher dimension space
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x) # x shape: [batch_size, embed_dim, num_patches_height, num_patches_width]
        x = x.flatten(2) # x shape: [batch_size, embed_dim, num_patches] Merge the height and width into a single sequence
        x = x.transpose(1, 2) # x shape: [batch_size, num_patches, embed_dim]

        return x

'''
Multi-head self attention: this step extract the image feature by finding the weight between different patches. 
Multi-head: Each heads has a set of qkv. 
self-attention: computing the "relationships" between patches. 
q, k, v = query, key, value
query: 
key: 
value: 
alpha = q x k; initial embedding step
softmax(alpha); get the attention weight
alpha x v; feature extraction
'''

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__() 
        self.num_heads = num_heads 
        self.embed_dim = embed_dim 
        self.head_dim = embed_dim // num_heads 

        self.qkv = nn.Linear(embed_dim, embed_dim * 3) 
        self.attn_drop = nn.Dropout(0.1)               # Why dropout? 
        self.proj = nn.Linear(embed_dim, embed_dim)    # Projection

    def forward(self, x, pos_embedding):

        B, N, C = x.shape # x shape: [batch_size, num_patches, embed_dim], B, N, C
        # qkv: query, key, value;  first convert to embedding
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2] 

        # Generate the trainable weights
        attn_weights = (q @ k.transpose(-2, -1)+pos_embedding) / (self.head_dim ** 0.5) 
        # softmax: 
        # converting the raw attention scores into a probability distribution ~ input patches. Amplify the relationship 
        # stabilizing gradients: smooth derivative 
        attn_weights = attn_weights.softmax(dim=-1) 
        attn_weights = self.attn_drop(attn_weights) 

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

"""
Encoder Layer: 
input: Patch + Position Embedding; Output: 

"""

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attention = Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 num_classes=10, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 max_point_num=128, 
                 mlp_ratio=4.0, 
                 feature=KF.KeyNetAffNetHardNet,
                 device=torch.device("cuda")):
        super().__init__()
        # Change embedding here
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.max_point_num = max_point_num

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embed = RelativeAttentionBias(embed_dim, max_point_num)

        self.pos_drop = nn.Dropout(0.1)
        
        self.encoder_layers = nn.ModuleList(
            [ TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio) for _ in range(depth) ]  # depth: how many layers the ViT should have. 
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

        self.feature = feature(max_point_num, True).eval().to(device)

    def forward(self, x):
        lafs, resps, descs = self.feature(K.color.rgb_to_grayscale(x))
        positions = K.feature.laf.get_laf_center(lafs)
        temp_eig,temp_V,temp_angle=get_laf_scale_and_angle(lafs)

        positions = torch.cat([positions, temp_eig, temp_angle], dim=-1)
        
        x = get_resized_patch(x,temp_angle[:,:,0],
                              temp_eig[:,:,0]/2, 
                              temp_eig[:,:,1]/2, 
                              size=(self.patch_size, self.patch_size))

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        pos_embedding = self.pos_embed(positions)
        for layer in self.encoder_layers:
            x = layer(x,pos_embedding)
        x = self.norm(x)

        return self.head(x[:, 0])
