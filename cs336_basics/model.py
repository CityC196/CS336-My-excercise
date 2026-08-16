import math
import torch
from torch import nn
from einops import einsum, rearrange

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

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model:int,
        eps:float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
        torch.ones(
            d_model,
            device=device,
            dtype=dtype,
            )
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = torch.mean(
            x**2,
            dim=-1,
            keepdim=True
        )
        ans = x / torch.sqrt(mean_square + self.eps) *self.weight
        return ans.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int,
        d_ff:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )
    def forward(self, x:torch.Tensor) ->torch.Tensor:
        w1_x = self.w1(x)
        w3_x = self.w3(x) #self.w3会直接利用call自动搜索forward
        silu_output = w1_x * torch.sigmoid(w1_x)
        gated_output = silu_output * w3_x
        output = self.w2(gated_output)
        return output

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )
        inverse_frequencies = theta ** (-dimension_indices / d_k)
        token_positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )
        rotation_angles = einsum(
            token_positions,
            inverse_frequencies,
            "position, dimension -> position dimension",
        )

        self.register_buffer(
            "cos_values",
            torch.cos(rotation_angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_values",
            torch.sin(rotation_angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        cos_values = self.cos_values[token_positions]
        sin_values = self.sin_values[token_positions]

        while cos_values.ndim < x.ndim:
            cos_values = cos_values.unsqueeze(-3)
            sin_values = sin_values.unsqueeze(-3)

        cos_values = cos_values.to(dtype=x.dtype)
        sin_values = sin_values.to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos_values - x_odd * sin_values
        rotated_odd = x_even * sin_values + x_odd * cos_values

        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd
        return output

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    maximum_values = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    shifted_x = x - maximum_values

    exp_values = torch.exp(shifted_x)
    exp_sum = torch.sum(
        exp_values,
        dim=dim,
        keepdim=True,
    )
    output = exp_values / exp_sum
    return output


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]

    attention_scores = einsum(
        Q,
        K,
        "... queries d_k, ... keys d_k -> ... queries keys",
    )
    scaled_attention_scores = attention_scores / math.sqrt(d_k)

    if mask is not None:
        additive_mask = torch.zeros_like(scaled_attention_scores)
        additive_mask = additive_mask.masked_fill(
            ~mask,
            float("-inf"),
        ) #运算技巧
        scaled_attention_scores = scaled_attention_scores + additive_mask

    attention_weights = softmax(
        x=scaled_attention_scores,
        dim=-1,
    )

    output = einsum(
        attention_weights,
        V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()


        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if theta is None and max_seq_len is None:
            self.rope = None
        elif theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = rearrange(
            queries,
            "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",
            num_heads=self.num_heads,
        )
        keys = rearrange(
            keys,
            "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",
            num_heads=self.num_heads,
        )
        values = rearrange(
            values,
            "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v",
            num_heads=self.num_heads,
        )

        sequence_length = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                )
            queries = self.rope(
                queries,
                token_positions,
            )
            keys = self.rope(
                keys,
                token_positions,
            )

        sequence_positions = torch.arange(
            sequence_length,
            device=x.device,
        )
        query_positions = rearrange(
            sequence_positions,
            "query -> query 1",
        )
        key_positions = rearrange(
            sequence_positions,
            "key -> 1 key",
        )
        causal_mask = query_positions >= key_positions

        attention_output = scaled_dot_product_attention(
            Q=queries,
            K=keys,
            V=values,
            mask=causal_mask,
        )
        attention_output = rearrange(
            attention_output,
            "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)",
        )

        output = self.output_proj(attention_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )
        x = x + self.ffn(self.ln2(x))
        return x
