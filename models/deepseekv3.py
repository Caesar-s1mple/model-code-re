import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch import Tensor
from .utils import Config


class DeepSeekV3(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.layers = nn.ModuleList([DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)])

        self.norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.max_seq_len = -1
        self.causal_mask = None
        self.freq_cis = None
        self.use_cache = False

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        actual_vocab_size = self.word_embeddings.weight.shape[0]
        old_embeddings = state_dict[prefix + 'word_embeddings.weight']
        if old_embeddings.shape[0] > actual_vocab_size:
            state_dict[prefix + 'word_embeddings.weight'] = old_embeddings[:actual_vocab_size, :].clone()

        old_linear_w = state_dict[prefix + 'linear.weight']
        if old_linear_w.shape[0] > actual_vocab_size:
            state_dict[prefix + 'linear.weight'] = old_linear_w[:actual_vocab_size, :].clone()

    def setup_caches(self, max_seq_len: int, use_cache=False):
        self.max_seq_len = max_seq_len
        dtype = self.linear.weight.dtype

        if hasattr(self.linear, 'scales'):
            dtype = self.linear.scales.dtype
        elif hasattr(self.linear, 'scales_and_zeros'):
            dtype = self.linear.scales_and_zeros.dtype

        self.use_cache = use_cache
        for layer in self.layers:
            if self.use_cache:
                layer.self_attn.kv_cache = KVCache()
            else:
                layer.self_attn.kv_cache = None

        self.freq_cis = precompute_freqs_cis(self.max_seq_len,
                                             self.config.embedding_dim // self.config.num_heads,
                                             self.config.rope_base,
                                             dtype,
                                             self.config.rope_scaling)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        bs, seq_len = input_ids.shape
        pre_len = 0
        if self.use_cache and self.layers[0].self_attn.kv_cache.k_cache is not None:
            pre_len = self.layers[0].self_attn.kv_cache.k_cache.size(2)
            input_ids = input_ids[:, pre_len:]

        causal_mask_2d = self.causal_mask[pre_len: seq_len, : seq_len]
        freqs_cis = self.freq_cis[pre_len: seq_len]

        if attention_mask is None:
            causal_mask_2d = causal_mask_2d.unsqueeze(0).expand(bs, -1, -1)
        else:
            attention_mask = attention_mask.unsqueeze(1).bool()
            attention_mask = attention_mask & attention_mask.transpose(1, 2)

            diag_indices = torch.arange(seq_len, device=input_ids.device)
            attention_mask[:, diag_indices, diag_indices] = True
            causal_mask_2d = causal_mask_2d.unsqueeze(0).expand(bs, -1, -1)
            causal_mask_2d = causal_mask_2d & attention_mask

        x = self.word_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, causal_mask_2d, freqs_cis)
        x = self.norm(x)
        logits = self.linear(x)

        return logits


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config)

        if layer_id >= config.first_moe_layer:
            self.ff = MoELayer(config)
        else:
            self.ff = FeedForward(config, config.feedforward_dim)

        self.attn_norm = RMSNorm(config.embedding_dim, config.norm_eps)
        self.ff_norm = RMSNorm(config.feedforward_dim, config.norm_eps)

    def forward(self, x: Tensor, causal_mask_2d: Tensor, freq_cis: Tensor):
        h = x + self.self_attn(self.attn_norm(x), causal_mask_2d, freq_cis)
        output = h + self.ff(self.ff_norm(h))

        return output


class FeedForward(nn.Module):
    def __init__(self, config, feedforward_dim):
        super().__init__()
        self.linear1 = nn.Linear(config.embedding_dim, feedforward_dim, bias=False)
        self.linear3 = nn.Linear(config.embedding_dim, feedforward_dim, bias=False)
        self.linear2 = nn.Linear(feedforward_dim, config.embedding_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor):
        return self.linear2(self.act(self.linear1(x)) * self.linear3(x))


class MoERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_num_experts = config.moe_num_experts
        self.moe_scaling_factor = config.moe_scaling_factor
        self.moe_num_groups = config.moe_groups
        self.moe_topk_groups = config.moe_topk_groups
        self.moe_topk_experts = config.moe_topk_experts

        assert self.moe_num_experts % self.moe_num_groups == 0, "num_routed_experts must be divisible by moe_num_groups"
        assert self.moe_topk_groups <= self.moe_num_groups, "moe_topk_groups must be less than or equal to moe_num_groups"
        assert self.moe_topk_experts <= self.moe_topk_groups * (self.num_routed_experts // self.moe_num_groups), \
            "moe_topk_experts must be less than or equal to moe_topk_groups * (moe_num_experts // moe_num_groups)"

        self.weight = nn.Parameter(torch.empty((self.moe_num_experts, self.embedding_dim)))
        self.register_buffer('e_score_correction_bias', torch.zeros(self.moe_num_experts))

    @torch.no_grad()
    def get_topk_ids(self, scores: Tensor):
        bs, seq_len = scores.shape[:2]
        scores_for_choice = scores.view(-1, self.moe_num_experts) + self.e_score_correction_bias.unsqueeze(0)  # bs * seq_len, moe_num_experts
        group_scores = scores_for_choice.view(-1, self.moe_group, self.moe_num_experts // self.moe_group).topk(2, dim=-1)[0].sum(dim=-1)  # bs * seq_len, moe_num_groups
        group_ids = torch.topk(group_scores, self.moe_topk_group, dim=-1, sorted=False)[1]  # bs * seq_len, moe_topk_groups
        group_mask = torch.zeros_like(group_scores)  # bs * seq_len, moe_num_groups
        group_mask.scatter_(dim=-1, index=group_ids, value=1)  # bs * seq_len, moe_num_groups
        score_mask = group_mask.unsqueeze(-1).expand(-1, self.moe_group, self.moe_num_experts // self.moe_group).reshape(-1, self.moe_num_experts)  # bs * seq_len, moe_num_experts

        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.)
        topk_ids = torch.topk(scores_for_choice, self.moe_topk_experts, dim=-1, sorted=False)[1]  # bs * seq_len, moe_topk_experts

        return topk_ids.view(bs, seq_len, self.moe_topk_experts)

    def forward(self, x: Tensor):
        # x -> bs, seq_len, embedding_dim
        router_logits = F.linear(x.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()  # bs, seq_len, moe_num_experts
        topk_ids = self.get_topk_ids(scores)  # bs, seq_len, moe_topk_experts
        topk_weights = scores.gather(-1, topk_ids)

        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.moe_scaling_factor
        return topk_ids, topk_weights


class MoELayer(nn.Module):
    def __init(self, config):
        super().__init__()
        self.moe_num_experts = config.moe_num_experts
        self.experts = nn.ModuleList([FeedForward(config, config.moe_feedforward_dim) for _ in range(config.moe_num_experts)])
        self.router = MoERouter(config)
        self.shared_experts = FeedForward(config, config.moe_feedforward_dim * config.moe_num_shared_experts)

    def forward(self, x: Tensor):
        topk_ids, topk_weights = self.router(x)
        # bs, seq_len, moe_topk_experts

        final_x = torch.zeros_like(x, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_ids, num_classes=self.moe_num_experts).permute(3, 0, 1, 2)  # moe_num_experts, bs, seq_len, moe_topk_experts

        for expert_idx in range(self.moe_num_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx] # bs, seq_len, moe_topk_experts
            batch_ids, seq_ids, topk_ids = torch.where(mask)
            if batch_ids.numel() > 0:
                expert_weights = topk_weights[batch_ids, seq_ids, topk_ids]  # num_selected
                expert_x = x[batch_ids, seq_ids]  # num_selected, embedding_dim
                expert_x = expert(expert_x)  # num_selected, embedding_dim
                expert_x = expert_x * expert_weights.unsqueeze(-1)
                final_x[batch_ids, seq_ids] += expert_x

        x = final_x.type_as(x) + self.shared_experts(x)

        return x


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.linear_q_a = nn.Linear(self.embedding_dim, self                                                                                                                                                                                                                     .q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
        self.linear_q_b = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.linear_kv_a = nn.Linear(self.embedding_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, config.norm_eps)
        self.linear_kv_b = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.linear = nn.Linear(self.num_heads * self.v_head_dim, self.embedding_dim, bias=False)

        self.kv_cache = None

    def forward(self, x: Tensor, causal_mask_2d: Tensor, freqs_cis: Tensor):
        bs, seq_len = x.shape[:2]

        query = self.q_a_norm(self.linear_q_a(x))
        query = self.linear_q_b(query)
        query = query.view(bs, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        query_n_rot, query_rot = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        query_rot = apply_rotate_emb(query_rot, freqs_cis)

        compressed_key_value, key_rot = self.linear_kv_a(x).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_key_value = self.kv_a_norm(compressed_key_value)

        key_rot = key_rot.view(bs, 1, seq_len, self.qk_rope_head_dim)
        key_rot = apply_rotate_emb(key_rot, freqs_cis)

        if self.kv_cache:
            key_rot, compressed_key_value = self.kv_cache.update(key_rot, compressed_key_value)

        compressed_key_value = self.linear_kv_b(compressed_key_value)
        compressed_key_value = compressed_key_value.view(bs, seq_len, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)

        key_n_rot, value = compressed_key_value.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        key_rot = key_rot.repeat_interleave(self.num_kv_heads, dim=1)

        query = torch.cat([query_n_rot, query_rot], dim=-1)
        key = torch.cat([key_n_rot, key_rot], dim=-1)

        key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attention_score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention_score.masked_fill_(~causal_mask_2d.unsqueeze(1), -torch.inf)

        attention_score = torch.softmax(attention_score, dim=-1, dtype=torch.float32).type_as(query)
        output = attention_score @ value
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.embedding_dim)

        output = self.linear(output)

        return output


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, norm_eps):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def _norm(self, x):
        # x -> bs, seq_len, embedding_dim
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.to(torch.float32)).type_as(x)
        return output * self.weight


class KVCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_rot_cache: Optional[Tensor] = None
        self.lora_kv_cache: Optional[Tensor] = None

    def update(self, k_rot_val: Tensor, lora_kv_val: Tensor):
        # k_rot_val -> bs, 1, seq_len, qk_rope_head_dim
        # lora_kv_val -> bs, seq_len, kv_lora_rank
        if self.k_rot_cache is None:
            self.k_rot_cache = k_rot_val
            self.lora_kv_cache = lora_kv_val
        else:
            self.k_rot_cache = torch.cat([self.k_cache, k_rot_val], dim=-2)
            self.lora_kv_cache = torch.cat([self.v_cache, lora_kv_val], dim=-2)

        return self.k_cache, self.v_cache


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


def precompute_freqs_cis(max_seq_len: int, dim: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16,
                         rope_scaling: Optional[dict] = None) -> Tensor:
    inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
    if rope_scaling is not None:
        inv_freq = apply_rope_scaling(inv_freq, rope_scaling)
    position = torch.arange(0, max_seq_len, dtype=torch.float)
    # position -> max_seq_len, 1  inv_freq -> dim / 2
    freqs = torch.outer(position, inv_freq)
    freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return freqs_cis.to(dtype=dtype)  # max_seq_len, dim // 2, 2


def apply_rotate_emb(x: Tensor, freqs_cis):
    # x -> bs, num_heads, seq_len, dim
    # freqs_cis -> seq_len, dim // 2, 2
    # 正常的权重转换而来的模型位置编码按两两一对一对的方式嵌入
    # x1 = x[..., ::2]
    # x2 = x[..., 1::2]
    # Huggingface权重转换而来的模型位置编码按前后对半方式嵌入
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    cos = freqs_cis[..., 0]  # seq_len, dim // 2
    sin = freqs_cis[..., 1]  # seq_len, dim // 2

    x_out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    return x_out.type_as(x)
