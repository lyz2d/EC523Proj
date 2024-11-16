import torch
import torch.nn as nn
import torch.nn.functional as F

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # generate token  t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar documents
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # The result is a comlex vector
    # if freqs = [x, y]
    # then freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # to complex
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # rotate and turn to real domain
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# The RoPE module
class laf_RoPE(nn.Module):
    """
    dim: int, the dimension of the embedding
    thetas: list[float], the hyperparameters for the positional encoding

    The out put dimension is BxLx5dim
    """
    def __init__(self, dim, 
                 seq_len=512,
                 thetas=[10000.0, 10000.0, 10000.0, 10000.0, 10000.0]):
        super().__init__()
        self.dim = 256
        self.thetas = thetas
        self.seq_len = seq_len

        self.freqs_cis_x = precompute_freqs_cis(dim, seq_len*2, theta=thetas[0])
        self.freqs_cis_y = precompute_freqs_cis(dim, seq_len*2, theta=thetas[1])
        self.freqs_cis_sx = precompute_freqs_cis(dim, seq_len*2, theta=thetas[2])
        self.freqs_cis_sy = precompute_freqs_cis(dim, seq_len*2, theta=thetas[3])
        self.freqs_cis_ox = precompute_freqs_cis(dim, seq_len*2, theta=thetas[4])

    def forward(self, 
                position: torch.Tensor, # BxLx2
                scale: torch.Tensor, # BxLx2
                orientation: torch.Tensor, #BxL
    ) -> torch.Tensor:
        x_=position[:,0].reshape(position.shao[:-1], -1, 2)
        y_=position[:,1].reshape(position.shao[:-1], -1, 2)
        sx_=scale[:,0].reshape(position.shao[:-1], -1, 2)
        sy_=scale[:,1].reshape(position.shao[:-1], -1, 2)
        ox_=orientation.reshape(position.shao[:-1], -1, 2)
        
        x_=torch.view_as_complex(x_)
        y_=torch.view_as_complex(y_)
        sx_=torch.view_as_complex(sx_)
        sy_=torch.view_as_complex(sy_)
        ox_=torch.view_as_complex(ox_)

        # rotate and turn to real domain
        x_out = torch.view_as_real(x_ * self.freqs_cis()).flatten(2)
        y_out = torch.view_as_real(y_ * self.freqs_cis()).flatten(2)
        sx_out = torch.view_as_real(sx_ * self.freqs_cis()).flatten(2)
        sy_out = torch.view_as_real(sy_ * self.freqs_cis()).flatten(2)
        ox_out = torch.view_as_real(ox_ * self.freqs_cis()).flatten(2)
        

        pos_embed = torch.cat([x_out, y_out, sx_out, sy_out, ox_out], dim=-1)
        return pos_embed
        




class SineCosine2DPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(SineCosine2DPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        # Calculate dimension for each axis (split equally between height and width)
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        self.height_dim = embed_dim // 2
        self.width_dim = embed_dim // 2

        # Create positional encodings
        self.register_buffer("positional_encoding", self.create_positional_encoding())

    def create_positional_encoding(self):
        # Create positional encoding for height (y-axis)
        y_pos = torch.arange(self.height, dtype=torch.float32).unsqueeze(1)  # Shape (H, 1)
        div_term_y = torch.exp(torch.arange(0, self.height_dim, 2) * -torch.log(torch.tensor(10000.0)) / self.height_dim)
        pos_y = torch.zeros(self.height, self.height_dim)
        pos_y[:, 0::2] = torch.sin(y_pos * div_term_y)
        pos_y[:, 1::2] = torch.cos(y_pos * div_term_y)

        # Create positional encoding for width (x-axis)
        x_pos = torch.arange(self.width, dtype=torch.float32).unsqueeze(1)  # Shape (W, 1)
        div_term_x = torch.exp(torch.arange(0, self.width_dim, 2) * -torch.log(torch.tensor(10000.0)) / self.width_dim)
        pos_x = torch.zeros(self.width, self.width_dim)
        pos_x[:, 0::2] = torch.sin(x_pos * div_term_x)
        pos_x[:, 1::2] = torch.cos(x_pos * div_term_x)

        # Combine height and width encodings
        pos_y = pos_y.unsqueeze(1).repeat(1, self.width, 1)  # Shape (H, W, D/2)
        pos_x = pos_x.unsqueeze(0).repeat(self.height, 1, 1)  # Shape (H, W, D/2)

        # Concatenate to get final positional encoding of shape (H, W, D)
        positional_encoding = torch.cat([pos_y, pos_x], dim=-1)
        return positional_encoding

    def forward(self, x):
        # x shape is (batch_size, height, width, embed_dim)
        batch_size, height, width, embed_dim = x.shape
        assert height == self.height and width == self.width, "Input dimensions must match positional embedding dimensions."

        # Add positional encoding to input
        return self.positional_encoding.unsqueeze(0).to(x.device)
    
if __name__ == "__main__":
    batch_size = 2
    height, width = 128, 128
    embed_dim = 64

    x = torch.randn(batch_size, height, width, embed_dim)
    pe=SineCosine2DPositionalEmbedding(embed_dim, height, width)
    pos_embed = pe(x)
    output = pe(x)
    import matplotlib.pyplot as plt
    print(output.shape)  # Expected shape: (2, 8, 8, 64)
    plt.imshow(output[0, :, 0, :].detach().numpy())
