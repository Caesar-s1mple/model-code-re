import json

import torch
import torch.nn as nn
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, \
    sample, norm_logits
from typing import Optional
from torch import Tensor
from pathlib import Path
from transformers import AutoTokenizer


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if 'cuda' in device:
        torch.cuda.synchronize(device)


def load_model(config_path: Path, checkpoint_path: Path, quantize: Optional[str] = None, device: str = default_device):
    config = Config(config_path)
    with torch.device('meta'):
        model = model_map[config.architecture](config)

    if quantize == 'int8':
        quantizer = Int8QuantHandler(model)
        model = quantizer.convert_for_runtime()
    elif quantize == 'int4':
        quantizer = WeightOnlyInt4QuantHandler(model)
        model = quantizer.convert_for_runtime()

    checkpoint = torch.load(checkpoint_path, mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model.eval()


def decode_one_token(model: nn.Module, prompt: Tensor, temperature: float, top_k: int, top_p: float):
    logits = model(prompt)
    probs = norm_logits(logits[:, -1:, :], temperature, top_k=top_k, top_p=top_p)
    next_token = sample(probs)

    return next_token


@torch.no_grad()
def generate(model: nn.Module, prompt: Tensor, max_new_tokens: int, eos_id: int, use_cache: bool = False, temperature: float = 1., top_k: int = 0,
             top_p: float = 0.):
    T = prompt.size(-1)
    T_new = T + max_new_tokens

    device = prompt.device
    with torch.device(device):
        model.setup_caches(max_seq_len=min(T_new, model.config.max_seq_len), use_cache=use_cache)

    for i in range(max_new_tokens):
        if prompt[0, -1] == eos_id:
            break
        next_token = decode_one_token(model, prompt, temperature, top_k, top_p)
        prompt = torch.cat([prompt, next_token], dim=-1)

    return prompt


def main(max_new_tokens: int, config_path: Path, checkpoint_path: Path, quantize: Optional[str] = None, use_cache: bool = False, device: str = default_device, temperature: float = 1., top_k: int = 0,
         top_p: float = 0.):
    model = load_model(config_path, checkpoint_path, quantize, device)
    device_sync(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
    original_data_list = []
    with open('./benchmark/redpajama/truncate_en_text.jsonl', encoding='utf-8') as f:
        for line in f.readlines():
            original_data_list.append(json.loads(line)['text'])

    with open('./benchmark/redpajama/msts_generation_text.jsonl', 'a', encoding='utf-8') as f:
        generation_list = []
        for text in original_data_list:
            input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
            length = input_ids.size(1)
            device_sync(device)
            output_ids = generate(
                model,
                prompt=input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
                use_cache=use_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            device_sync(device)
            generation_text = tokenizer.decode(output_ids[0, length:], skip_special_tokens=True)
            generation_list.append(generation_text)
            f.write(json.dumps({
                'prefix': text,
                'generation': generation_text
            }, ensure_ascii=False) + '\n')
