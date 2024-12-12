# Modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py


import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Grayscale

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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



# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# traditional 2D sinusoidal positional embedding.
# It is not used in the current implementation, but it can be a choice.
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    


class SIFT_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.sift_predict = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 5),
            nn.Tanh()
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.laf_extractor = LafPatchExtractor(patch_size=patch_height)
        self.cov_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, padding=0)

        x_num = image_width // patch_width
        y_num = image_height // patch_height
        self.x_num = x_num
        self.y_num = y_num
        self.point_num = x_num*y_num
        X0,Y0= torch.meshgrid(torch.arange(0.5,x_num), torch.arange(0.5, y_num))
        # calculate the position of the center of the patch in the image normalized coordinate
        X0=2*(X0/x_num)-1
        Y0=2*(Y0/y_num)-1
        self.register_buffer('X0', X0)
        self.register_buffer('Y0', Y0)

        self.alpha = nn.Parameter(torch.tensor(0.005))
        

    # generate original laf
    # def original_laf(self, batch_size=1):
    #     lafs = torch.zeros(batch_size, self.point_num, 2, 3)  # (B, P, 2, 3)
    #     lafs[:, :, 0, 1] = torch.ones(batch_size, self.point_num)
    #     lafs[:, :, 1, 0] = torch.ones(batch_size, self.point_num)
    #     lafs[:, :, 0, 2] = self.X0.flatten()[None, :]
    #     lafs[:, :, 1, 2] = self.Y0.flatten()[None, :]
    #     return lafs


    # predict lafs from image
    def predict_lafs(self, img):
        pred_params = self.sift_predict(img) # (B, P, 5) tx, ty, sx, sy, theta
        params = self.alpha*pred_params+(1-self.alpha)*torch.zeros_like(pred_params)
        params[:, :, -1] = params[:, :, -1] * torch.pi
        lafs = torch.zeros(params.shape[0], params.shape[1], 2, 3, device=next(self.parameters()).device)  # (B, P, 2, 3)
        lafs[:, :, 0, 2] = params[:, :, 0]+self.X0.flatten()[None, :] # position shift
        lafs[:, :, 1, 2] = params[:, :, 1]+self.Y0.flatten()[None, :] # position shift
        lafs[:,:,0,0] = torch.cos(params[:,:,4])*params[:,:,2]
        lafs[:,:,0,1] = -torch.sin(params[:,:,4])*params[:,:,3]
        lafs[:,:,1,0] = torch.sin(params[:,:,4])*params[:,:,2]
        lafs[:,:,1,1] = torch.cos(params[:,:,4])*params[:,:,3]
        return lafs

        
    def forward(self, img):

        lafs = self.predict_lafs(img)
        x = self.laf_extractor(img, lafs)
        x = self.cov_embedding(x)
        x = x.flatten(2).transpose(1, 2)


        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    


class Ssd_ViT(nn.Module):
    def __init__(self, *, detector, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_size = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.x_num = image_width // patch_width
        self.y_num = image_height // patch_height

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.detector = detector
        self.grayscale = Grayscale(num_output_channels=1)

        W0,H0= torch.meshgrid(torch.arange(self.x_num)*self.patch_size, torch.arange(self.y_num)*self.patch_size) # P
        laf = torch.tensor([[[1,0,0],[0,1,0]]])
        self.lafs = laf.repeat(self.x_num*self.y_num, 1, 1) # (P, 2, 3)
        self.lafs[:,0,2] = W0.flatten()
        self.lafs[:,1,2] = H0.flatten() # (P, 2, 3)

        self.extractor = LafPatchExtractor(patch_size=patch_size)

    def forward(self, img):
        shifts = self.detect_shifts(img)
        patches = self.extract_patches(img, shifts) # patches is a big picture of patches

        # Covolutional embedding on the big picture to make patches into embeddings
        x = self.cov_embedding(patches)
        x = x.flatten(2).transpose(1, 2)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def detect_shifts(self, img):
        img = self.grayscale(img)
        respose = self.detector(img)
        respose_patch = self.to_patch(respose) # (B, P, 256)
        indice = respose_patch.argmax(dim=-1)
        shift_H = indice//self.patch_size # (B, P)
        shift_W = indice%self.patch_size # (B, P)
        return shift_W, shift_H
        

    def extract_patches(self, img, shifts):
        shift_W, shift_H = shifts
        padding = self.patch_size//2
        batched_laf = self.lafs.view(1,-1).repeat(img.shape[0], 1, 1, 1) # (B, P, 2, 3)
        batched_laf[:,:,0,2] = batched_laf[:,:,0,2] + shift_W + padding
        batched_laf[:,:,1,2] = batched_laf[:,:,1,2] + shift_H + padding
        img = F.pad(img, (padding,padding,padding,padding,), value=0)
        patches = self.extractor(img, batched_laf)

        return patches