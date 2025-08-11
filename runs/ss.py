import torch
import torch.nn as nn
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, sample, norm_logits, norm_max
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


def draft_one_token(model: nn.Module, prompt: Tensor, temperature: float, top_k: int, top_p: float):
    logits = model(prompt)
    probs = norm_logits(logits, temperature, top_k=top_k, top_p=top_p)
    next_token = sample(probs[:, -1, :])

    return next_token, probs


def verify_tokens(model: nn.Module, prompt: Tensor, temperature: float, top_k: int, top_p: float):
    logits = model(prompt)
    probs = norm_logits(logits, temperature, top_k=top_k, top_p=top_p)

    return probs


def rollback(model: nn.Module, prob_history, end_position: int):
    for layer in model.layers:
        k_cache = layer.self_attn.kv_cache.k_cache  # bs, num_kv_heads, cached_len, head_dim
        v_cache = layer.self_attn.kv_cache.v_cache
        k_cache = k_cache[:, :, :end_position + 1, :]
        v_cache = v_cache[:, :, :end_position + 1, :]
        layer.self_attn.kv_cache.k_cache = k_cache
        layer.self_attn.kv_cache.v_cache = v_cache

    prob_history = prob_history[:, :end_position + 1, :]
    return prob_history


@torch.no_grad()
def generate(target_model: nn.Module, draft_model: nn.Module, prompt: Tensor, max_new_tokens: int, gamma: int,
             eos_id: int, use_cache: bool = False, temperature: float = 1., top_k: int = 0,
             top_p: float = 0.):
    T = prompt.size(-1)
    T_new = T + max_new_tokens

    device = prompt.device
    with torch.device(device):
        target_model.setup_caches(max_seq_len=min(T_new + gamma - 1, target_model.config.max_seq_len), use_cache=use_cache)

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
            if use_cache:
                if draft_prob_history is None:
                    draft_prob_history = prob
                else:
                    draft_prob_history = torch.cat([draft_prob_history, prob], dim=1)
            else:
                draft_prob_history = prob
            prompt = torch.cat([prompt, next_token], dim=-1)

        prob = verify_tokens(target_model, prompt, temperature, top_k, top_p)
        if use_cache:
            if target_prob_history is None:
                target_prob_history = prob
            else:
                target_prob_history = torch.cat([target_prob_history, prob], dim=1)
        else:
            target_prob_history = prob

        n = prefix_len + gamma - 1
        for i in range(gamma):
            r = torch.rand(1, device=device)
            j = prompt[:, prefix_len + i]
            if r > (target_prob_history[:, prefix_len + i - 1, j] / draft_prob_history[:, prefix_len + i - 1, j]):
                n = prefix_len + i - 1
                break

            draft_token_cnt += 1
            if j == eos_id:
                n = prefix_len + i
                eos = True
                break

        prompt = prompt[:, :n + 1]
        if use_cache:
            draft_prob_history = rollback(draft_model, draft_prob_history, n)
        if eos:
            break
        if n < prefix_len + gamma - 1:
            t = sample(norm_max(target_prob_history[:, n, :], draft_prob_history[:, n, :]))
            if use_cache:
                target_prob_history = rollback(target_model, target_prob_history, n)
            resample_token_cnt += 1
        else:
            t = sample(target_prob_history[:, n, :])
            if use_cache:
                target_prob_history = rollback(target_model, target_prob_history, n + 1)
            target_token_cnt += 1

        prompt = torch.cat([prompt, t], dim=-1)
        if t == eos_id:
            break

    return prompt


def main(prompt: str, max_new_tokens: int, config_path: Path, checkpoint_path: Path, draft_config_path: Path,
         draft_checkpoint_path: Path, num_samples: int = 3,
         quantize: Optional[str] = None, draft_quantize: Optional[str] = None, gamma: int = 3, use_cache: bool = False,
         device: str = default_device, temperature: float = 1., top_k: int = 0,
         top_p: float = 0., dialogue: bool = False, system_prompt: str = 'You are a helpful assistant.'):
    target_model = load_model(config_path, checkpoint_path, quantize, device)
    draft_model = load_model(draft_config_path, draft_checkpoint_path, draft_quantize, device)

    assert target_model.config.vocab_size == draft_model.config.vocab_size, "vocab_size of target model and draft model do not match"

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
    for i in range(num_samples):
        device_sync(device)
        output_ids = generate(
            target_model,
            draft_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            eos_id=tokenizer.eos_token_id,
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        device_sync(device)
        output_text = tokenizer.decode(output_ids[0])
        outputs.append(output_text)
        print('-----------------------------------------')
        print(output_text)
        print('-----------------------------------------')

    return outputs
