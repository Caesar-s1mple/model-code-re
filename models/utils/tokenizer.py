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


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path, dialogue: bool = False, system_prompt: str = ''):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
                             "<|begin_of_text|>",
                             "<|end_of_text|>",
                             "<|reserved_special_token_0|>",
                             "<|reserved_special_token_1|>",
                             "<|reserved_special_token_2|>",
                             "<|reserved_special_token_3|>",
                             "<|start_header_id|>",
                             "<|end_header_id|>",
                             "<|reserved_special_token_4|>",
                             "<|eot_id|>",  # end of turn
                         ] + [
                             f"<|reserved_special_token_{i}|>"
                             for i in range(5, self.num_reserved_special_tokens - 5)
                         ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=model_path.name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]
        self._eot_id: int = self.special_tokens["<|eot_id|>"]

        self.dialogue = dialogue
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
        tokens = [self._bos_id] + self.model.encode(text, allowed_special='all')
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def eot_id(self):
        return self._eot_id


class TiktokenTokenizer(TokenizerInterface):
    def __init__(self, tokenizer_path: Path, dialogue=False, system_prompt=''):
        super().__init__(tokenizer_path)
        self.dialogue = dialogue
        self.system_prompt = system_prompt

        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        with open(tokenizer_path.parent / 'special_tokens_map.json', 'r', encoding='utf-8') as f:
            special_tokens_json = json.load(f)

        vocab = tokenizer_json['model']['vocab']
        merges = tokenizer_json['model']['merges']
        special_tokens = tokenizer_json.get('added_tokens', [])
        special_tokens_dict = {token['content']: token['id'] for token in special_tokens}

        mergable_ranks = {}
        for rank, merge in enumerate(merges):
            b1, b2 = merge.split()
            mergable_ranks[(b1.encode('utf-8'), b2.encode('utf-8'))] = rank

        vocab = {token: id for token, id in vocab.items() if token not in special_tokens_dict}
        pat_str = tokenizer_json['model'].get('pattern',
                                              r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.encoding = tiktoken.Encoding(
            name='custom_bpe',
            pat_str=pat_str,
            mergeable_ranks=mergable_ranks,
            special_tokens=special_tokens_dict
        )

        self._bos_id = vocab[special_tokens_json['bos_token']]
        self._eos_id = vocab[special_tokens_json['eos_token']]

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
        tokens = [self._bos_id] + self.encoding.encode(text)
        return torch.tensor(tokens, dtype=torch.int)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def get_tokenizer(tokenizer_model_path, model_name, dialogue: bool = False, system_prompt: str = ''):
    if "llama-3" in str(model_name).lower():
        return TiktokenWrapper(Path(tokenizer_model_path), dialogue, system_prompt)
    else:
        return SentencePieceWrapper(tokenizer_model_path, dialogue, system_prompt)
