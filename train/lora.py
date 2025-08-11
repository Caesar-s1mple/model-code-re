import torch
import torch.nn as nn
import torch.distributed as dist
import os
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, replace_linear_with_lora, LinearWithLoRA
from typing import Optional, List
from torch import Tensor
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from .utils.data_processor import CasualLLMDataset
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
               device: str, lora_rank: int, lora_alpha: float, lora_dropout: float, except_modules: Optional[List[str]] = None):
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
    replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, except_modules=except_modules)

    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    model = model.to(device=device, dtype=torch.bfloat16)
    return model


def train_epoch(model, dataloader, optimizer, device: str, gradient_accumulation_steps: int):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        step_loss = loss.item() * gradient_accumulation_steps
        total_loss += step_loss

    return total_loss / len(dataloader)


def merge_lora_weight(model):
    for name, child in list(model.named_children()):
        if isinstance(child, LinearWithLoRA):
            child.merge()
            setattr(model, name, child.linear)
        else:
            merge_lora_weight(child)


def main(config_path: Path, checkpoint_path: Path, train_file_path: Path, epochs: int, batch_size: int, lr: float,
         quantize: Optional[str] = None, max_length: int = 512, num_workers: int = 1, devices: Optional[List[str]] = None, dialogue: bool = False, gradient_accumulation_steps: int = 1,
         lora_rank: int = 8, lora_alpha: float = 32, lora_dropout: float = 0., except_modules: Optional[List[str]] = None):

    local_rank, world_size, device = setup_distributed(devices)

    model = load_model(config_path, checkpoint_path, quantize, device, lora_rank, lora_alpha, lora_dropout, except_modules)
    device_sync(device)
    model = DDP(model, device_ids=[int(device.split(':')[-1])], output_device=int(device.split(':')[-1]))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
    dataset = CasualLLMDataset(train_file_path, tokenizer, max_length, dialogue)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=True, num_workers=num_workers, collate_fn=dataset.collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"===== Epoch {epoch}/{epochs} =====")
        avg_loss = train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps)

        if local_rank == 0:
            print(f'Epoch {epoch}/{epochs} completed. Average loss: {avg_loss:.4f}')
            merge_lora_weight(model.module)
            torch.save(model.state_dict(), checkpoint_path.parent / (f'lora_e{epoch}bs{batch_size}lr{lr}gas{gradient_accumulation_steps}' + (checkpoint_path.name)))

    dist.destroy_process_group()
