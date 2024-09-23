import time
import torch
from models import LLaMA
from models.utils import Config, WeightOnlyInt8QuantHandler, get_tokenizer, set_seed, sample, norm_logits
from typing import Optional
from torch import Tensor

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def device_sync(device):
    if 'cuda' in device:
        torch.cuda.synchronize(device)


def model_forward(model: LLaMA, input_ids: Tensor, input_pos: Tensor):
    return model(input_ids, input_pos)


def prefill(model: LLaMA, input_ids: Tensor, input_pos: Tensor, temperature: float = 1, top_k: int = 0,
            top_p: float = 0.):
    logits = model(input_ids, input_pos)
    logits = norm_logits(logits, temperature, top_k, top_p)
    return sample(logits)


def decode_one_token(model: LLaMA, input_id: Tensor, input_pos: Tensor, temperature: float = 1, top_k: int = 0,
                     top_p: float = 0.):
    assert input_pos.size(-1) == 1
    logits = model(input_id, input_pos)
    logits = norm_logits(logits, temperature, top_k, top_p)
    return sample(logits)


@torch.no_grad()
def generate(model: LLaMA, prompt: Tensor, max_new_tokens: int, temperature: float = 1, top_k: int = 0,
             top_p: float = 0.):
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_len = min(T_new, model.config.max_seq_len)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_len=max_seq_len)

    seq = torch.empty(1, T_new, dtype=dtype, device=device)
    seq[:, :T] = prompt
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, temperature, top_k, top_p)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    new_tokens = []
    for i in range(max_new_tokens - 1):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_token(model, next_token.view(1, -1), input_pos, temperature, top_k, top_p)
            input_pos += 1
            new_tokens.append(next_token.clone())

    seq[:, T + 1:] = torch.cat(new_tokens, dim=-1)

    return seq


def load_model(config_path, checkpoint_path, quantize: Optional[str] = None, device: str = default_device):
    with torch.device('meta'):
        model = LLaMA(Config(config_path))

    if quantize == 'int8':
        int8_quantizer = WeightOnlyInt8QuantHandler(model)
        model = int8_quantizer.convert_for_runtime()

    checkpoint = torch.load(checkpoint_path, mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model.eval()


def main(prompt: str, max_new_tokens: int, config_path: str, checkpoint_path: str, num_samples: int = 1,
         quantize: Optional[str] = None, device: str = default_device):
    model = load_model(config_path, checkpoint_path, quantize, device)

    device_sync(device)

    tokenizer = get_tokenizer('../models/checkpoints/Meta-Llama-3-8B/original/tokenizer.model', model_name='llama-3')
    prompt = tokenizer.encode(prompt)

    set_seed(20010101)

    global decode_one_token
    decode_one_token = torch.compile(decode_one_token, mode='reduce-overhead', fullgraph=True)

    outputs = []
    for i in range(-1, num_samples):
        device_sync(device)
        t0 = time.perf_counter()
        output_ids = generate(
            model,
            prompt,
            max_new_tokens
        )
        if i == -1:
            print(f'Compilation time: {time.perf_counter() - t0:.2f} seconds')
            continue
        device_sync(device)
        output_text = tokenizer.decode(output_ids)
        outputs.append(output_text)
        print(output_text)

    return outputs


if __name__ == '__main__':
    main(prompt='Hello, my name is',
         max_new_tokens=500,
         config_path='../models/config/llama-3-8b_compile.json',
         checkpoint_path='../models/checkpoints/Meta-Llama-3-8B/convert/model.pth',
         num_samples=5,
         quantize='int8',
         device='cuda:0')
