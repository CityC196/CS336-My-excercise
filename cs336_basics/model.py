import math
import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device:torch.device | None = None,
        dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )
        std = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(
            self.weight, 0.0, std, -3*std, 3*std
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight , "... input, output input -> ... output")

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings:int,
        embedding_dim:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight,0 ,1,-3,3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  self.weight[x]
