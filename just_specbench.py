import time
import argparse
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer
from pathlib import Path
import json
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, sample, norm_logits, norm_max
from typing import Optional
from models.utils import set_seed

default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
set_seed(123456)


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
def generate(model: nn.Module, prompt: Tensor, max_new_tokens: int, eos_id: int, use_cache: bool = False,
             temperature: float = 1., top_k: int = 0, top_p: float = 0.):
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


@torch.no_grad()
def generate_ss(target_model: nn.Module, draft_model: nn.Module, prompt: Tensor, max_new_tokens: int, gamma: int,
                eos_id: int, use_cache: bool = False, temperature: float = 1., top_k: int = 0, top_p: float = 0.):
    T = prompt.size(-1)
    T_new = T + max_new_tokens

    device = prompt.device
    with torch.device(device):
        target_model.setup_caches(max_seq_len=min(T_new + gamma - 1, target_model.config.max_seq_len),
                                  use_cache=use_cache)

    with torch.device(device):
        draft_model.setup_caches(max_seq_len=min(T_new + gamma - 1, draft_model.config.max_seq_len),
                                 use_cache=use_cache)

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

    return prompt, draft_token_cnt, target_token_cnt, resample_token_cnt


def main_ar():
    model = load_model(Path(config_path), Path(checkpoint_path), quantize, device)
    device_sync(device)

    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint_path).parent)

    spec_data = []
    with open('./benchmark/specbench.jsonl', encoding='utf-8') as f:
        for line in f.readlines():
            spec_data.append(json.loads(line))

    total_time = 0.
    total_token_cnt = 0
    turn_cnt = 0
    for i, data in enumerate(spec_data):
        messages = []
        turn_time = 0.
        turn_token_cnt = 0
        for turn in data['turns']:
            turn_cnt += 1
            messages.append({'role': 'user', 'content': turn})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt = tokenizer([prompt], return_tensors='pt')['input_ids'].to(device)

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
            turn_time += time.time() - t0
            total_time += turn_time
            device_sync(device)
            generated_ids = output_ids[0][len(prompt[0]):]
            turn_token_cnt += len(generated_ids)
            total_token_cnt += turn_token_cnt
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({'role': 'assistant', 'content': output_text})
            # print(messages)
        print(f'{data["question_id"]} Averaged token time: {turn_time:.2f}/{turn_token_cnt}={turn_time / turn_token_cnt:.6f}')
        with open(output_record_path, 'a', encoding='utf-8') as f:
            f.write(f"{data['question_id']},{data['category']},{len(data['turns'])},{turn_token_cnt},{turn_time}\n")

    print(f'Averaged token time: {total_time:.2f}/{total_token_cnt}={total_time / total_token_cnt:.6f}')


def main_ss():
    target_model = load_model(Path(config_path), Path(checkpoint_path), quantize, device)
    draft_model = load_model(Path(draft_config_path), Path(draft_checkpoint_path), draft_quantize, device)
    device_sync(device)

    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint_path).parent)

    spec_data = []
    with open('./benchmark/specbench.jsonl', encoding='utf-8') as f:
        for line in f.readlines():
            spec_data.append(json.loads(line))

    total_time = 0.
    total_token_cnt = 0
    turn_cnt = 0
    for i, data in enumerate(spec_data):
        messages = []
        turn_time = 0.
        turn_token_cnt = 0
        draft_token_cnt = 0
        target_token_cnt = 0
        resample_token_cnt = 0
        for turn in data['turns']:
            turn_cnt += 1
            messages.append({'role': 'user', 'content': turn})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt = tokenizer([prompt], return_tensors='pt')['input_ids'].to(device)

            t0 = time.time()
            output_ids, _draft_token_cnt, _target_token_cnt, _resample_token_cnt = generate_ss(
                target_model,
                draft_model,
                gamma=gamma,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
                use_cache=use_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            turn_time += time.time() - t0
            total_time += turn_time
            device_sync(device)
            generated_ids = output_ids[0][len(prompt[0]):]
            turn_token_cnt += len(generated_ids)
            total_token_cnt += turn_token_cnt
            draft_token_cnt += _draft_token_cnt
            target_token_cnt += _target_token_cnt
            resample_token_cnt += _resample_token_cnt
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({'role': 'assistant', 'content': output_text})

        print(f'{data["question_id"]} Averaged token time: {turn_time:.2f}/{turn_token_cnt}={turn_time / turn_token_cnt:.6f} Draft token count: {draft_token_cnt} Target token count: {target_token_cnt} Resample token count: {resample_token_cnt}')
        with open(output_record_path, 'a', encoding='utf-8') as f:
            f.write(f"{data['question_id']},{data['category']},{len(data['turns'])},{turn_token_cnt},{draft_token_cnt},{target_token_cnt},{resample_token_cnt},{turn_time}\n")

    print(f'Averaged token time: {total_time:.2f}/{total_token_cnt}={total_time / total_token_cnt:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='ar', choices=['ar', 'ss'])
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--config_path', type=str, default='./models/config/llama-2-7b.json')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/llama-7b/convert/model.pth')
    parser.add_argument('--quantize', type=str, default=None)

    parser.add_argument('--draft_config_path', type=str, default='./models/config/llama-2-160m.json')
    parser.add_argument('--draft_checkpoint_path', type=str, default='./checkpoints/llama-160m/convert/model.pth')
    parser.add_argument('--draft_quantize', type=str, default=None)
    parser.add_argument('--gamma', type=int, default=16)

    parser.add_argument('--exp_name', type=str, default='deepseek')

    args = parser.parse_args()

    output_record_path = './msts/{}-{}.csv'.format(args.strategy, args.exp_name)

    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    device = args.device
    use_cache = args.use_cache
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    quantize = args.quantize
    draft_config_path = args.draft_config_path
    draft_checkpoint_path = args.draft_checkpoint_path
    draft_quantize = args.draft_quantize
    gamma = args.gamma

    if args.strategy == 'ar':
        with open(output_record_path, 'a+', encoding='utf-8') as f:
            f.write("question_id,category,turn_num,generate_token_count,time\n")
        main_ar()
    elif args.strategy == 'ss':
        with open(output_record_path, 'a+', encoding='utf-8') as f:
            f.write("question_id,category,turn_num,generate_token_count,draft_token_count,target_token_count,resample_token_count,time\n")
        main_ss()
