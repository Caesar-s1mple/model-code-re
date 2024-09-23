import torch
import torch.nn as nn
import math
from typing import Optional
from torch import Tensor
from .utils import Config


class LLaMA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.max_batch_size = -1
        self.max_seq_len = -1
        self.freqs_cis = None
        self.mask_cache = None
        self.causal_mask = None

    def setup_caches(self, max_batch_size: int, max_seq_len: int):
        if self.max_batch_size >= max_batch_size and self.max_seq_len >= max_seq_len:
            return

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        head_dim = self.config.embedding_dim // self.config.num_heads
        dtype = self.linear.weight.dtype
        for layer in self.layers:
            layer.self_attn.kv_cache = KVCache(max_batch_size, max_seq_len, self.config.num_kv_heads, head_dim,
                                               dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.max_seq_len,
                                              self.config.embedding_dim // self.config.num_heads, self.config.rope_base,
                                              dtype, self.config.rope_scaling)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))

    def forward(self, input_ids: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        attention_mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]

        x = self.word_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask, freqs_cis, input_pos)
        x = self.norm(x)
        logits = self.linear(x)

        return logits


class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.self_attn = GroupedMultiQueryAttention(config)
        self.ff = FeedForward(config)
        self.ff_norm = RMSNorm(config)
        self.attn_norm = RMSNorm(config)

    def forward(self, x: Tensor, attention_mask: Tensor, freq_cis: Tensor, input_pos: Tensor) -> Tensor:
        h = x + self.self_attn(self.attn_norm(x), attention_mask, freq_cis, input_pos)
        output = h + self.ff(self.ff_norm(h))

        return output


class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"
        self.num_kv_heads = config.num_kv_heads

        self.linear_qkv = nn.Linear(self.embedding_dim, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                                    bias=False)
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.kv_cache = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "linear_q.weight" in state_dict:
            wq = state_dict.pop(prefix + "linear_q.weight")
            wk = state_dict.pop(prefix + "linear_k.weight")
            wv = state_dict.pop(prefix + "linear_v.weight")
            state_dict[prefix + "linear_qkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, attention_mask: Tensor, freqs_cis: Tensor, input_pos: Optional[Tensor] = None):
        bs = x.size(0)

        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = self.linear_qkv(x).split([self.head_dim, kv_size, kv_size], dim=-1)

        query = query.view(bs, -1, self.num_heads, self.head_dim)
        key = key.view(bs, -1, self.num_kv_heads, self.head_dim)
        value = value.view(bs, -1, self.num_kv_heads, self.head_dim)

        query = apply_rotary_embed(query, freqs_cis)
        key = apply_rotary_embed(key, freqs_cis)

        query, key, value = map(lambda x: x.transpose(1, 2), (query, key, value))

        if self.kv_cache is not None:
            key, value = self.kv_cache.update(key, value, input_pos)

        key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=-1)
        value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=-1)

        attention_score = query @ key.transpose(-2, -1) / math.sqrt(self.embedding_dim)
        if attention_mask is not None:
            attention_score.masked_fill_(attention_mask.unsqueeze(1) == 0, -torch.inf)

        attention_score = torch.softmax(attention_score, dim=-1)
        output = attention_score @ value
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.embedding_dim)

        output = self.fc(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim, bias=False),
            nn.SiLU()
        )
        self.linear2 = nn.Linear(config.embedding_dim, config.feedforward_dim, bias=False)
        self.linear3 = nn.Linear(config.feedforward_dim, config.embedding_dim, bias=False)

    def forward(self, x: Tensor):
        return self.linear3(self.linear1(x) * self.linear2(x))


class RMSNorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.embedding_dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight


class KVCache(nn.Module):
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, k_val, v_val, input_pos):
        assert input_pos.size(0) == k_val.size(2)
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


def apply_rotary_embed(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # bs, seq_len, num_heads, dim / 2, 2
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # 1, seq_len, 1, dim / 2, 2
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def apply_rope_scaling(freqs: Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(seq_len: int, dim: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16,
                         rope_scaling: Optional[dict] = None) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # seq_len, dim
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)  # max_seq_len, dim // 2, 2
