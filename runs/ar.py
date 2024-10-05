import torch
import torch.nn as nn
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, get_tokenizer, \
    sample, norm_logits
from typing import Optional
from torch import Tensor
from pathlib import Path


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
    logits = norm_logits(logits[:, -1:, :], temperature, top_k=top_k, top_p=top_p)
    next_token = sample(logits)

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
         top_p: float = 0., dialogue: bool = False, system_prompt: str = 'You are a helpful AI assistant'):
    model = load_model(config_path, checkpoint_path, quantize, device)

    device_sync(device)

    tokenizer = get_tokenizer(checkpoint_path.parent / 'tokenizer.model', model_name='llama-3', dialogue=dialogue, system_prompt=system_prompt)
    prompt = tokenizer.encode(prompt).to(device).view(1, -1)

    outputs = []
    for i in range(num_samples):
        device_sync(device)
        output_ids = generate(
            model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eot_id() if dialogue else tokenizer.eos_id(),
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        device_sync(device)
        output_text = tokenizer.decode(output_ids[0].tolist())
        outputs.append(output_text)
        print('-----------------------------------------')
        print(output_text)
        print('-----------------------------------------')

    return outputs
