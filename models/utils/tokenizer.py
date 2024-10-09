import json
import os
import sentencepiece as spm
import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe
from typing import Dict
from pathlib import Path


class TokenizerInterface:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text, bos=True):
        tokens = self.processor.EncodeAsIds(text)
        if bos:
            tokens = [self.bos_id()] + tokens
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()


class TiktokenTokenizer(TokenizerInterface):
    def __init__(self, tokenizer_path: Path, dialogue=False, system_prompt=''):
        super().__init__(tokenizer_path)
        self.dialogue = dialogue
        self.system_prompt = system_prompt

        with open(tokenizer_path / 'tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        with open(tokenizer_path / 'special_tokens_map.json', 'r', encoding='utf-8') as f:
            special_tokens_json = json.load(f)

        vocab = tokenizer_json['model']['vocab']
        special_tokens = tokenizer_json.get('added_tokens')
        special_tokens_dict = {token['content']: token['id'] for token in special_tokens}

        char_to_byte = bytes_to_unicode()
        mergeable_ranks = {}
        for token, token_id in vocab.items():
            if token not in special_tokens_dict:
                byte_sequence = bytes([char_to_byte[c] for c in token])
                mergeable_ranks[byte_sequence] = int(token_id)
        pat_str = tokenizer_json['model'].get('pattern',
                                              r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

        self.encoding = tiktoken.Encoding(
            name='custom_bpe',
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_dict
        )

        self._bos_id = special_tokens_dict[special_tokens_json['bos_token']]
        self._eos_id = special_tokens_dict[special_tokens_json['eos_token']]
        self._eot_id = special_tokens_dict.get('<|eot_id|>', None)

        if self.dialogue:
            if len(system_prompt) == 0:
                system_prompt = 'You are a helpful AI assistant'
            self.template = '<|start_header_id|>system<|end_header_id|>\n\n' \
                            + system_prompt \
                            + '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n' \
                            + '{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

    def encode(self, text):
        if self.dialogue:
            text = self.template.format(text)
        tokens = [self._bos_id] + self.encoding.encode(text, allowed_special='all')
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def eot_id(self):
        return self._eot_id


def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1))  # 33-126
    bs += list(range(ord('¡'), ord('¬') + 1))  # 161-172
    bs += list(range(ord('®'), ord('ÿ') + 1))  # 174-255
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 将Unicode码点转换为字符
    cs = [chr(c) for c in cs]
    return dict(zip(cs, bs))


def get_tokenizer(tokenizer_model_path, dialogue: bool = False, system_prompt: str = ''):
    return TiktokenTokenizer(Path(tokenizer_model_path), dialogue, system_prompt)
    # else:
    #     return SentencePieceWrapper(tokenizer_model_path, dialogue, system_prompt)
