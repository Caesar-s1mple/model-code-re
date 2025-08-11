import torch
import torch.nn as nn
import math


class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        attention_mask = attention_mask.unsqueeze(1).expand(attention_mask.size(0), attention_mask.size(1),
                                                            attention_mask.size(1))

        attention_scores = []
        output = embeddings
        for layer in self.encoder_layers:
            output, attention_score = layer(output, attention_mask)
            attention_scores.append(attention_score)

        logits = self.linear(output)

        return logits, attention_scores


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim,
                                            padding_idx=config.tokenizer['<pad>'])
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embedding_dim)
        self.segment_embeddings = nn.Embedding(config.num_segments, config.embedding_dim)

        self.norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.embedding_dropout_rate)

        self.register_buffer('position_ids', torch.arange(config.max_seq_len).expand((1, -1)), persistent=False)
        self.register_buffer('segment_ids', torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(self, input_ids, position_ids=None, segment_ids=None):
        seq_len = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        if segment_ids is None:
            buffered_segment_ids = self.segment_ids[:, :seq_len].expand(input_ids.size(0), seq_len)
            segment_ids = buffered_segment_ids

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = word_embeddings + position_embeddings + segment_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # torch里是有multi-head attention的，但这里为了学习需要还是手写吧。ps: torch里的linear是带bias的，但原本论文里是不带的
        # self.self_attn = nn.MultiheadAttention(config.embedding_dim, config.num_heads)
        self.self_attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.embedding_dim)
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, encoder_input, attention_mask=None):
        output1, attention_score = self.self_attn(encoder_input, encoder_input, encoder_input, attention_mask)
        output1 = encoder_input + self.dropout(output1)
        output1 = self.norm1(output1)

        output2 = self.ff(output1)
        output2 = output1 + self.dropout(output2)
        output2 = self.norm2(output2)

        return output2, attention_score


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)  # (embed_dim, num_heads * head_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.ff = nn.Linear(self.embed_dim, self.embed_dim, bias=False)  # (num_heads * head_num, embed_dim)

    def forward(self, query, key, value, attention_mask=None):
        # attention_mask -> (bs, seq_len_1, seq_len_2)
        bs = query.size(0)

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        query = query.view(bs, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2).contiguous()  # (bs, num_heads, seq_len_1, head_dim)
        key = key.view(bs, -1, self.num_heads, self.head_dim).transpose(1,
                                                                        2).contiguous()  # (bs, num_heads, seq_len_2, head_dim)
        value = value.view(bs, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2).contiguous()  # (bs, num_heads, seq_len2, head_dim)

        attention_score = query @ key.transpose(-2, -1) / math.sqrt(self.embed_dim)  # (bs, num_heads, seq_len_1, seq_len_2)
        if attention_mask is not None:
            attention_score.masked_fill_(attention_mask.unsqueeze(1) == 0, -torch.inf)

        attention_score = torch.softmax(attention_score, dim=-1)
        output = attention_score @ value  # (bs, num_heads, seq_len_1, head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)

        output = self.ff(output)

        return output, attention_score


class SegmentEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

