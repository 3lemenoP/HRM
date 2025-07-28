from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
    USE_FLASH_ATTN = True
except ImportError:
    try:
        # Fallback to FlashAttention 2
        from flash_attn import flash_attn_func  # type: ignore[import]
        USE_FLASH_ATTN = True
    except ImportError:
        # Use PyTorch's built-in attention as final fallback
        print("Warning: Flash Attention not found. Using PyTorch's built-in attention (slower but functional).")
        USE_FLASH_ATTN = False
        
        # Define dummy function to avoid import errors
        def flash_attn_func(*args, **kwargs):
            raise NotImplementedError("Flash Attention not available")

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if USE_FLASH_ATTN:
            # Use Flash Attention
            attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
            if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
                attn_output = attn_output[0]
        else:
            # Fallback to PyTorch's built-in attention
            # Reshape for PyTorch attention: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
            query = query.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Create attention mask if causal
            attn_mask = None
            if self.causal:
                attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
            
            # Use PyTorch's scaled dot product attention
            attn_output = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask, 
                dropout_p=0.0, 
                is_causal=False  # We handle causal with explicit mask
            )
            # Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
            attn_output = attn_output.transpose(1, 2)

        # attn_output: [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


# Gating mechanism implementation
class HRMGatingNetwork(nn.Module):
    """Core gating network for controlling information flow between HRM modules."""
    
    def __init__(self, hidden_size: int, num_gates: int = 3, gate_hidden_ratio: float = 0.25,
                 gate_init_bias: float = -2.2):
        super().__init__()
        # Lightweight design: use fraction of hidden size for gate computation
        gate_hidden = int(hidden_size * gate_hidden_ratio)
        
        # Two-layer MLP for gate computation
        self.gate_proj = nn.Sequential(
            CastedLinear(hidden_size * 3, gate_hidden, bias=True),
            nn.SiLU(),  # Smooth activation for gates
            CastedLinear(gate_hidden, num_gates * hidden_size, bias=True)
        )
        
        # Initialize gates to be slightly open
        with torch.no_grad():
            self.gate_proj[-1].bias.fill_(gate_init_bias)
    
    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor, 
                x_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gates for L, H, and input channels."""
        # Concatenate all inputs for gate computation
        gate_input = torch.cat([z_L, z_H, x_input], dim=-1)
        
        # Compute gates through MLP
        gates = self.gate_proj(gate_input)
        
        # Reshape to separate gates
        gates = gates.view(*gates.shape[:-1], 3, -1)
        
        # Apply sigmoid for [0, 1] range
        gates = torch.sigmoid(gates)
        
        # Split into individual gates
        gate_L, gate_H, gate_X = gates.unbind(dim=-2)
        
        return gate_L, gate_H, gate_X
