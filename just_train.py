import argparse
from pathlib import Path
from train import full, lora, msts
from models.utils import set_seed
import os
import torch
import torch.multiprocessing as mp

set_seed(123456)


def main_worker(local_rank, world_size, devices, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)

    if args.strategy == 'full':
        full.main(config_path=Path(args.config_path),
                  checkpoint_path=Path(args.checkpoint_path),
                  train_file_path=Path(args.train_file_path),
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  quantize=args.quantize,
                  max_length=args.max_length,
                  num_workers=args.num_workers,
                  devices=devices,
                  dialogue=args.dialogue,
                  gradient_accumulation_steps=args.gradient_accumulation_steps
                  )
    elif args.strategy == 'lora':
        lora.main(config_path=Path(args.config_path),
                  checkpoint_path=Path(args.checkpoint_path),
                  train_file_path=Path(args.train_file_path),
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  quantize=args.quantize,
                  max_length=args.max_length,
                  num_workers=args.num_workers,
                  devices=devices,
                  dialogue=args.dialogue,
                  gradient_accumulation_steps=args.gradient_accumulation_steps,
                  lora_rank=args.rank,
                  lora_alpha=args.alpha,
                  except_modules=args.exclude_modules
                  )
    elif args.strategy == 'msts':
        msts.main(config_path=Path(args.config_path),
                  checkpoint_path=Path(args.checkpoint_path),
                  train_file_path=Path(args.train_file_path),
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  quantize=args.quantize,
                  max_length=args.max_length,
                  num_workers=args.num_workers,
                  devices=devices,
                  dialogue=args.dialogue,
                  gradient_accumulation_steps=args.gradient_accumulation_steps,
                  speculative_steps=args.speculative_steps,
                  temperatures=args.temperatures,
                  prob_path=Path(args.prob_path)
                  )


def spawn_main(args):
    if args.devices is None:
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices = args.devices

    world_size = len(devices)
    print(f" [INFO] Spawning {world_size} processes for DDP...")

    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(world_size, devices, args),
        join=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 训练策略
    parser.add_argument('--strategy', type=str, default='full', choices=['full', 'lora', 'msts'])
    # 常规设置
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--devices', type=str, nargs='+', default=None)
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    # 模型指定
    parser.add_argument('--config_path', type=str, default='./models/config/llama-2-7b.json')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/llama-7b/convert/model.pth')
    parser.add_argument('--train_file_path', type=str,
                        default='./benchmark/redpajama/msts_generation_text_llama-2-7b.jsonl')
    parser.add_argument('--quantize', type=str, default=None, choices=[None, 'int8', 'int4'])
    parser.add_argument('--dialogue', type=bool, default=False)
    parser.add_argument('--system_prompt', type=str, default='You are a helpful assistant.')
    # LoRA设置(strategy='lora')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=32)
    parser.add_argument('--exclude_modules', type=str, nargs='+', default=['linear.weight'])
    # MSTS设置(strategy='msts')
    parser.add_argument('--speculative_steps', type=int, default=9)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument('--prob_path', type=str, default='./msts/llama-2-7b-ss9.pth')

    args = parser.parse_args()

    spawn_main(args)
