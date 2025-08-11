import torch.nn as nn
import torch
import math


def apply_rotary_emb(x, freqs_cis):
    # freqs_cis -> seq_len, head_dim // 2, 2
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    cos = freqs_cis[..., 0]
    sin = freqs_cis[..., 1]

    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x1 * sin + x2 * cos
    x_rotated = torch.cat([x1_rotated, x2_rotated], dim=-1)

    return x_rotated


class GroupedMultiHeadAttention(nn.Module):
    def __init(self, config):
        super().__init()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads

        self.linear_qkv = nn.Linear(self.embedding_dim, self.embedding_dim + 2 * self.kv_dim)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, x, causal_mask, freqs_cis):
        bs, seq_len = x.shape[:2]
        q, k, v = self.linear_qkv(x).split([self.embedding_dim, self.kv_dim, self.kv_dim], dim=-1)
        # q -> bs, seq_len, embedding_dim
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        k = k.repeat_interleave(self.num_heads // self.kv_dim, dim=1)
        v = v.repeat_interleave(self.num_heads // self.kv_dim, dim=1)

        attention_score = q @ k.transpose(1, 2) / math.sqrt(self.head_dim)
        attention_score.masked_fill_(~causal_mask, -torch.inf)

        attention_score = torch.softmax(attention_score, dim=-1)
        output = attention_score @ v
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.embedding_dim)
        output = self.linear(output)

        return output, attention_score


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
