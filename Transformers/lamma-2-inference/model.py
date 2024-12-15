import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional



@dataclass
class ModelArgs:
    dim: int  = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int , seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # x shape -> B, seq_len, H, head_dim, x is already devided into H heads
    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim)
    x_rotated = x_complex * freqs_complex
    return torch.view_as_real(x_rotated).view(*x.shape).type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int):
    b, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(b, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(b, seq_len, n_kv_heads * n_rep, head_dim)
        )



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def _norm(self, x):
        # x: B, seq_len, head_dim
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x).type_as(x) * self.weight

class SqiGlu(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        pass
    def forward(self, x):
        pass

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        return self.w2(x)

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads_q = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # indicates  how many times the heads of the Keys and Values should be repeated to match th head og the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q * args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * args.dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * args.dim, bias=False)
        # projection matrix
        self.wo = nn.Linear(self.n_heads_q * args.dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        B, seq_len, _ = x.shape # (B, 1, dim)
        # B, 1, dim -> B, 1, n_Q * h_dim
        xq = self.wq(x)
        # B, 1, dim -> B, 1, n_KV * h_dim
        xv = self.wv(x)
        # B, 1, dim -> B, 1, n_KV * h_dim
        xk = self.wk(x)

        xq = xq.view(B, seq_len, self.n_heads_q,  self.head_dim)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_dim)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, xq.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, xk.device)

        self.cache_k[:B, start_pos:start_pos + seq_len] = xk
        self.cache_v[:B, start_pos:start_pos + seq_len] = xv
        # retrieve cached
        keys = self.cache_k[:B, :start_pos + seq_len]
        values = self.cache_v[:B, :start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2,)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq)

        out = torch.matmul(scores, values)

        out = (out.transpose(1, 2).contiguous().view(b, seq_len, -1))

        return self.wo(out)

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        x = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        x = x + self.feed_forward(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set!"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.argmax_seq_len * 2, device=self.args.device)


    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (b, seq_len)
        B, T = tokens.shape
        assert T == 1, "Only one token as a time can be processed"
        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos:start_pos + T]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h)
        return output

































