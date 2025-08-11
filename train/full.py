import torch
import torch.nn as nn
import torch.distributed as dist
import os
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler
from typing import Optional, List
from torch import Tensor
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from .utils.data_processor import CasualLLMDataset
from torch.optim import AdamW
import deepspeed


def load_model(config_path: Path, checkpoint_path: Path, quantize: Optional[str]):
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

    return model


def train_epoch(model_engine, dataloader, loss_fn):
    model_engine.train()
    total_loss = 0.

    for batch in dataloader:
        input_ids = batch['input_ids'].to(model_engine.local_rank)
        attention_mask = batch['attention_mask'].to(model_engine.local_rank)
        labels = batch['labels'].to(model_engine.local_rank)

        logits = model_engine(input_ids, attention_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        model_engine.backward(loss)
        model_engine.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(config_path: Path, checkpoint_path: Path, train_file_path: Path, epochs: int, batch_size: int, lr: float,
         quantize: Optional[str] = None, max_length: int = 512, num_workers: int = 1, dialogue: bool = False):

    model = load_model(config_path, checkpoint_path, quantize)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.parent)
    dataset = CasualLLMDataset(train_file_path, tokenizer, max_length, dialogue)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=dataset.collate_fn)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model_engine, _, dataloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config="ds_config.json"
    )

    for epoch in range(1, epochs + 1):
        if model_engine.local_rank == 0:
            print(f"===== Epoch {epoch}/{epochs} =====")
        avg_loss = train_epoch(model_engine, dataloader, loss_fn)

        if model_engine.local_rank == 0:
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            save_path = checkpoint_path.parent / f'e{epoch}bs{batch_size}lr{lr}_' + checkpoint_path.name
            model_engine.save_checkpoint(save_path)
