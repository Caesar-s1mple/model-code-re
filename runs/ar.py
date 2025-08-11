import time
import torch
import torch.nn as nn
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, sample, norm_logits
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


def main(prompt: str, max_new_tokens: int, config_path: Path, checkpoint_path: Path, num_samples: int = 3,
         quantize: Optional[str] = None, use_cache: bool = False, device: str = default_device, temperature: float = 1., top_k: int = 0,
         top_p: float = 0., dialogue: bool = False, system_prompt: str = ''):
    model = load_model(config_path, checkpoint_path, quantize, device)
    device_sync(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
    if dialogue:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    prompt = tokenizer([prompt], return_tensors='pt')['input_ids'].to(device)

    outputs = []
    total_time = 0.
    for i in range(num_samples):
        device_sync(device)
        t0 = time.time()
        output_ids = generate(
            model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_token_id,
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        t1 = time.time()
        total_time += t1 - t0
        device_sync(device)
        output_text = tokenizer.decode(output_ids[0])
        outputs.append(output_text)
        print(f'-----Generation {i + 1}---Used time:{t1 - t0:.2f}-----')
        print(output_text)
        print('-----------------------------------------')

    print(f'Averaged used time: {total_time / num_samples:.2f}')

    return outputs
