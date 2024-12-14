# Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py


import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from SIFT.get_patch_and_feature import *

class LafPatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super(LafPatchExtractor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x, laf):
        '''
        x: (B, C, H, W)
        laf: (B, P, 2, 3)
        '''
        B, C, H_in, W_in = x.shape
        P = laf.shape[1]

        # laf from (B, P, 2, 3) to (B*P, 2, 3)
        laf = laf.view(-1, 2, 3)

        grid = F.affine_grid(laf, torch.Size((B*P, C, self.patch_size, self.patch_size)), align_corners=False)

        # grid to (B, P*patch_size, patch_size, 2) 
        grid = grid.view(B, P*self.patch_size, self.patch_size, 2)

        # Extract patch from image. Note the patch is aligned in row as a big picture.
        x = F.grid_sample(x, grid, mode='bilinear',align_corners=True) #x: (B, C, P*patch_size, patch_size) 
        return x

class LafPredictor(nn.Module):
    def __init__(self, patch_count=14):
        super(LafPredictor, self).__init__()
        
        
    def forward(self, img):
        pass

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,num_patches=64):
        super().__init__()
        self.num_patches=num_patches
        self.patch_size=patch_size
        patch_dim = in_channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
    def forward(self, x, batch_laf):  # the input is batch of img, tensor of shape (B,3,H,W)
        # batch_laf = get_lafs_for_batch_images(x,max_point_num=self.num_patches)
        x = get_patches_for_batch_images(x,batch_laf,size_resize=[self.patch_size,self.patch_size], max_point_num=self.num_patches)
        x = rearrange(x, 'b n c h1 w1 -> b n (c h1 w1)')
        x = self.to_patch_embedding(x) 

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self,embed_dim=5):
        super().__init__()
        
        feature_dim = 6 # (scale: 2,angle:1 ,center:2)
        self.to_Positional_embedding = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, batch_laf):  # the input is batch of img, tensor of shape (B,3,H,W)
        # scale,angle,center=get_feature_from_LAF(batch_laf)
        # # x=torch.cat((scale,angle,center), -1)
        # x=torch.cat((scale,center), -1)
        x = rearrange(batch_laf, 'b n h1 w1 -> b n (h1 w1)')
        x=self.to_Positional_embedding(x) 
        return x

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,num_patches):
        super().__init__()
        # Change embedding here
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels=3, embed_dim=dim,num_patches=num_patches)##########
        self.num_patches = num_patches
        self.pos_embed = PositionalEmbedding(embed_dim = dim)###########
        # self.max_point_num = max_point_num
        self.patch_size = patch_size
        self.embed_dim = dim





        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.pos_embedding2 = nn.Parameter(torch.randn(1, 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img,laf):
        device = img.device


        # laf = get_lafs_for_batch_images(img,max_point_num=self.num_patches) # num_patch = 

        x = self.patch_embed(img,laf)




        # x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        cls_tokens = cls_tokens + self.pos_embedding2[:, :].clone()



        


        y = self.pos_embed(laf)
        x = x+y

        x = torch.cat((cls_tokens, x), dim=1)
        # y=torch.cat((self.pos_embedding2, y), dim=1)


    

        # x += self.pos_embedding[:, :(n + 1)]


        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
