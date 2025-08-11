import random

import torch
import torch.nn as nn
import torch.distributed as dist
import os
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, get_tokenizer, \
    sample, norm_logits, norm_max
from typing import Optional, List
from torch import Tensor
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from .utils.data_processor import MSTSDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW


def setup_distributed(devices: Optional[List[str]] = None):
    dist.init_process_group(backend='nccl')
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    if devices:
        device = devices[local_rank]
    else:
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    torch.cuda.set_device(device)

    return local_rank, world_size, device


def device_sync(device):
    if 'cuda' in str(device):
        torch.cuda.synchronize(device)


def load_model(config_path: Path, checkpoint_path: Path, quantize: Optional[str],
               device: str):
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
    return model


def train_epoch(model, dataloader, optimizer, device: str, gradient_accumulation_steps: int, speculative_steps: int,
                temperatures: List[float], clip: float):
    model.train()
    total_loss = 0.

    g = 0.
    z_loss = torch.full((len(temperatures),), 0.0, device=device)
    for step, batch in enumerate(dataloader):
        g += 1
        idx = random.randint(0, len(temperatures) - 1)
        Tj = temperatures[idx]
        bs, max_length = batch['input_ids'].shape
        prob_tensor = batch['probs']

        b = torch.full((bs, max_length), 0.0, device=device)
        for i in range(max_length):
            b[:, i] = prob_tensor[:, i: min(i + speculative_steps + 1, max_length + 1)].sum(dim=1)
        b = (1 / bs) * ((1 - Tj) / Tj) * b

        w = torch.exp(torch.clip(((1 - Tj) / Tj) * (prob_tensor - b), None, clip))  # bs, max_length

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        logits = torch.log(norm_logits(logits, Tj))

        for i in range(max_length):
            logits[:, i] = w[:, i] * logits[:, i: min(i + speculative_steps + 1, max_length + 1)].sum(dim=1)

        valid_mask = (labels != -100).float()

        masked_logits = logits * valid_mask
        loss = masked_logits.sum() / valid_mask.sum()
        z_loss[idx] += loss.detach()

        loss = (1. / (z_loss[idx] / g)) * loss
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        step_loss = loss.item() * gradient_accumulation_steps
        total_loss += step_loss

    return total_loss / len(dataloader)


def main(config_path: Path, checkpoint_path: Path, train_file_path: Path, epochs: int, batch_size: int, lr: float,
         speculative_steps: int, temperatures: List[float], prob_path: Path, quantize: Optional[str] = None,
         max_length: int = 512, num_workers: int = 1, devices: Optional[List[str]] = None, dialogue: bool = False,
         gradient_accumulation_steps: int = 1):
    local_rank, world_size, device = setup_distributed(devices)

    model = load_model(config_path, checkpoint_path, quantize, device)
    device_sync(device)
    model = DDP(model, device_ids=[int(device.split(':')[-1])], output_device=int(device.split(':')[-1]))

    prob_tensor = torch.load(prob_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
    dataset = MSTSDataset(train_file_path, prob_path, tokenizer, max_length)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=True, num_workers=num_workers,
                            collate_fn=dataset.collate_fn, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"===== Epoch {epoch}/{epochs} =====")
        avg_loss = train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps, speculative_steps,
                               temperatures, prob_tensor)

        if local_rank == 0:
            print(f'Epoch {epoch}/{epochs} completed. Average loss: {avg_loss:.4f}')
            torch.save(model.state_dict(), checkpoint_path.parent / (
                        f'e{epoch}bs{batch_size}lr{lr}gas{gradient_accumulation_steps}' + (checkpoint_path.name)))

    dist.destroy_process_group()
