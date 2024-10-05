import torch
import torch.nn as nn
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, get_tokenizer, \
    sample, norm_logits, norm_max
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


def draft_one_token(model: nn.Module, prompt: Tensor, temperature: float, top_k: int, top_p: float):
    logits = model(prompt)
    logits = norm_logits(logits, temperature, top_k=top_k, top_p=top_p)
    next_token = sample(logits[:, -1, :])

    return next_token, logits


def verify_tokens(model: nn.Module, prompt: Tensor, temperature: float, top_k: int, top_p: float):
    logits = model(prompt)
    logits = norm_logits(logits, temperature, top_k=top_k, top_p=top_p)

    return logits


@torch.no_grad()
def generate(target_model: nn.Module, draft_model: nn.Module, prompt: Tensor, max_new_tokens: int, gamma: int,
             eos_id: int, use_cache: bool = False, temperature: float = 1., top_k: int = 0,
             top_p: float = 0.):
    T = prompt.size(-1)
    T_new = T + max_new_tokens

    device = prompt.device
    with torch.device(device):
        target_model.setup_caches(max_seq_len=min(T_new, target_model.config.max_seq_len), use_cache=use_cache)

    with torch.device(device):
        draft_model.setup_caches(max_seq_len=min(T_new + gamma - 1, draft_model.config.max_seq_len), use_cache=use_cache)

    target_prob_history: Optional[Tensor] = None
    draft_prob_history: Optional[Tensor] = None

    eos = False
    draft_token_cnt = 0
    target_token_cnt = 0
    resample_token_cnt = 0
    while prompt.size(-1) < T_new:
        prefix_len = prompt.size(-1)
        for _ in range(gamma):
            next_token, prob = draft_one_token(draft_model, prompt, temperature, top_k, top_p)
            if draft_prob_history is None:
                draft_prob_history = prob
            else:
                draft_prob_history = torch.cat([draft_prob_history, prob], dim=1)
            prompt = torch.cat([prompt, next_token], dim=-1)

        prob = verify_tokens(target_model, prompt, temperature, top_k, top_p)
        if target_prob_history is None:
            target_prob_history = prob
        else:
            target_prob_history = torch.cat([target_prob_history, prob], dim=1)

        n = prefix_len + gamma - 1
        for i in range(gamma):
            r = torch.rand(1, device=device)
            j = prompt[:, prefix_len + i]
            if r > (target_prob_history[:, prefix_len + i - 1, j]) / (draft_prob_history[:, prefix_len + i - 1, j]):
                n = prefix_len + i - 1
                break

            draft_token_cnt += 1
            if j == eos_id:
                n = prefix_len + i
                eos = T
                break

        prompt = prompt[:, :n + 1]
        draft_model.rollback(n)
        if eos:
            break
        if n < prefix_len + gamma - 1:
            t = sample(norm_max(target_prob_history[:, n, :], draft_prob_history[:, n, :]))
            target_model.rollback(n)
            resample_token_cnt += 1
        else:
            t = sample(target_prob_history[:, -1, :])
            target_model.rollback(n + 1)
            target_token_cnt += 1

        prompt = torch.cat([prompt, t], dim=-1)
        if t == eos_id:
            break

    return prompt


def main(prompt: str, max_new_tokens: int, config_path: Path, checkpoint_path: Path, draft_config_path: Path,
         draft_checkpoint_path: Path, num_samples: int = 3,
         quantize: Optional[str] = None, draft_quantize: Optional[str] = None, gamma: int = 3, use_cache: bool = False,
         device: str = default_device, temperature: float = 1., top_k: int = 0,
         top_p: float = 0., dialogue: bool = False):
    target_model = load_model(config_path, checkpoint_path, quantize, device)
    draft_model = load_model(draft_config_path, draft_checkpoint_path, draft_quantize, device)

    assert target_model.config.vocab_size == draft_model.config.vocab_size, "vocab_size of target model and draft model do not match"

    device_sync(device)

    tokenizer = get_tokenizer(checkpoint_path.parent / 'tokenizer.model', model_name='llama-3')
    prompt = tokenizer.encode(prompt).to(device).view(1, -1)

    outputs = []
    for i in range(num_samples):
        device_sync(device)
        output_ids = generate(
            target_model,
            draft_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            eos_id=tokenizer.eos_id(),
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        device_sync(device)
        output_text = tokenizer.decode(output_ids[0].tolist())
        outputs.append(output_text)
        print(output_text)

    return outputs


if __name__ == '__main__':
    main(prompt='Hello, my name is',
         max_new_tokens=200,
         config_path=Path('../models/config/llama-3-8b.json'),
         checkpoint_path=Path('../checkpoints/Meta-Llama-3-8B/convert/model_int8.pth'),
         num_samples=5,
         quantize='int8',
         use_cache=True,
         device='cuda:0')
