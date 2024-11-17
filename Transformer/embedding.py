import torch
import torch.nn as nn
import torch.nn.functional as F






"""
Do not use the RoPE below. Not sure if PoPE fits the multiple dimension positional embedding

"""
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

        self.freqs_cis_x = precompute_freqs_cis(dim, seq_len, theta=thetas[0])
        self.freqs_cis_y = precompute_freqs_cis(dim, seq_len, theta=thetas[1])
        self.freqs_cis_sx = precompute_freqs_cis(dim, seq_len, theta=thetas[2])
        self.freqs_cis_sy = precompute_freqs_cis(dim, seq_len, theta=thetas[3])
        self.freqs_cis_ox = precompute_freqs_cis(dim, seq_len, theta=thetas[4])

    def forward(self, 
                position: torch.Tensor, # BxLx2
                scale: torch.Tensor, # BxLx2
                orientation: torch.Tensor, #BxL
    ) -> torch.Tensor:
        x_=position[:,0].reshape(position.shape[:-1], -1, 2)
        y_=position[:,1].reshape(position.shape[:-1], -1, 2)
        sx_=scale[:,0].reshape(position.shape[:-1], -1, 2)
        sy_=scale[:,1].reshape(position.shape[:-1], -1, 2)
        ox_=orientation.reshape(position.shape[:-1], -1, 2)
        
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
        
if __name__ == "__main__":
    batch_size = 2
    height, width = 128, 128
    embed_dim = 64

    pass
