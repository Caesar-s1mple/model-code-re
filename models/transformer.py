import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim,
                                            padding_idx=config.tokenizer['<pad>'])
        self.position_embeddings = PositionalEncoding(config)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, encoder_input_ids, decoder_input_ids, src_attention_mask=None, tgt_attention_mask=None):
        # 注意这里的src_attention_mask是一个(bs, src_seq_len)的矩阵，和huggingface的transformers库对齐，在应该层更方便实现
        # tgt_attention_mask为一个下三角矩阵（对角线也为1） -> (bs, tgt_seq_len, tgt_seq_len)
        # 上面提到的两个矩阵应该在模型之外得到并传入，src_attention_mask一般在tokenizer时一并得到，tgt_attention_mask调用generate_subsequent_attention_mask这个静态方法得到

        # 为啥要乘embed_dim？论文说的，我不道啊论文也没讲别问我
        encoder_embeddings = self.word_embeddings(encoder_input_ids) * math.sqrt(self.embeddings.embedding_dim)
        encoder_embeddings = self.position_embeddings(encoder_embeddings)

        decoder_embeddings = self.embeddings(decoder_input_ids) * math.sqrt(self.embeddings.embedding_dim)
        decoder_embeddings = self.positional_encodings(decoder_embeddings)

        # (bs, src_seq_len) -> (bs, src_seq_len, src_seq_len)
        src_attention_mask = src_attention_mask.unsqueeze(1).expand(src_attention_mask.size(0),
                                                                    src_attention_mask.size(1),
                                                                    src_attention_mask.size(1))
        # (bs, src_seq_len) -> (bs, tgt_seq_len, src_seq_len)
        memory_attention_mask = src_attention_mask.data.unsqueeze(1).expand(src_attention_mask.size(0),
                                                                            tgt_attention_mask.size(1),
                                                                            src_attention_mask.size(1))

        encoder_attention_scores = []
        encoder_output = encoder_embeddings
        for layer in self.encoder_layers:
            encoder_output, encoder_attention_score = layer(encoder_output, src_attention_mask)
            encoder_attention_scores.append(encoder_attention_score)

        decoder_attention_scores = []
        encoder_decoder_attention_scores = []
        decoder_output = decoder_embeddings
        for layer in self.decoder_layers:
            decoder_output, decoder_attention_score, encoder_decoder_attention_score = layer(decoder_output,
                                                                                             encoder_output,
                                                                                             tgt_attention_mask,
                                                                                             memory_attention_mask)
            decoder_attention_scores.append(decoder_attention_score)
            encoder_decoder_attention_scores.append(encoder_decoder_attention_score)

        logits = self.linear(decoder_output)

        return logits, encoder_attention_scores, decoder_attention_scores, encoder_decoder_attention_scores

    @staticmethod
    def generate_subsequent_attention_mask(sz, device='cpu'):
        # 这里和网上绝大部分代码实现不一样，我认为我的这个更具备一致性，和encoder的attention_mask对齐了
        return torch.tril(torch.ones((sz, sz), device=device))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # torch里是有multi-head attention的，但这里为了学习需要还是手写吧。ps: torch里的linear是带bias的，但原本论文里是不带的
        # self.self_attn = nn.MultiheadAttention(config.embedding_dim, config.num_heads)
        self.self_attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.ReLU(),
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


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiheadAttention(config)
        self.encoder_decoder_attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.ReLU(),
            nn.Linear(config.feedforward_dim, config.embedding_dim)
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.norm3 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, decoder_input, encoder_input, tgt_attention_mask=None, memory_attention_mask=None):
        # src_attention_mask -> (bs, target_seq_len, target_seq_len)
        # memory_attention_mask -> (bs, target_seq_len, encoder_seq_len)
        decoder_output1, decoder_attention_score = self.self_attn(decoder_input, decoder_input, decoder_input,
                                                                  tgt_attention_mask)
        decoder_output1 = decoder_input + self.dropout(decoder_output1)
        decoder_output1 = self.norm1(decoder_output1)

        decoder_output2, encoder_decoder_attention_score = self.encoder_decoder_attn(decoder_output1, encoder_input,
                                                                                     encoder_input,
                                                                                     memory_attention_mask)
        decoder_output2 = decoder_output1 + self.dropout(decoder_output2)
        decoder_output2 = self.norm2(decoder_output2)

        decoder_output3 = self.ff(decoder_output2)
        decoder_output3 = decoder_output2 + self.dropout(decoder_output3)
        decoder_output3 = self.norm3(decoder_output3)

        return decoder_output3, decoder_attention_score, encoder_decoder_attention_score


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)

        pe = torch.zeros(config.max_seq_len, config.embedding_dim)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        # 实际transformer代码实现中，这里一般用对数计算进行高效计算，这里的实现和论文对齐（后续会补充其他实现）
        div_term = 10000 ** (torch.arange(0, config.embedding_dim, 2).float() / config.embedding_dim)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x -> (bs, seq_len, embed_dim)
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


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

        attention_score = query @ key.transpose(-2, -1) / math.sqrt(
            self.embed_dim)  # (bs, num_heads, seq_len_1, seq_len_2)
        if attention_mask is not None:
            attention_score.masked_fill_(attention_mask.unsqueeze(1) == 0, -torch.inf)

        attention_score = torch.softmax(attention_score, dim=-1)
        output = attention_score @ value  # (bs, num_heads, seq_len_1, head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)

        output = self.ff(output)

        return output, attention_score
