import json
import os
import torch
from pathlib import Path
import re
import shutil
import argparse
from safetensors.torch import load_file


def convert(hf_repo_path: Path, save_path: Path, weight_map_path: Path):
    with open(weight_map_path, 'r', encoding='utf-8') as f:
        weight_map = json.load(f)
    original_weights_path = hf_repo_path / 'original' / 'consolidated.00.pth'
    if os.path.exists(original_weights_path):
        original_weights = torch.load(original_weights_path, map_location='cpu', weights_only=True)
    else:
        original_weights = {}
        for file in os.listdir(hf_repo_path):
            if file.endswith(('.bin', '.pth', 'pt')):
                weights = torch.load(hf_repo_path / file, map_location='cpu', weights_only=True)
                original_weights.update(weights)
            if file.endswith('safetensors'):
                weights = load_file(hf_repo_path / file, device='cpu')
                original_weights.update(weights)

    new_state_dict = {}
    for name, param in original_weights.items():
        if name in weight_map:
            new_name = weight_map[name]
            if '&&' in new_name:
                new_names = new_name.split('&&')
                for new_name in new_names:
                    if new_name.startswith('(r)'):
                        new_name = new_name[3:]
                        new_state_dict[new_name] = param
                    else:
                        new_state_dict[new_name] = param
            else:
                new_state_dict[new_name] = param
        else:
            abstract_name = re.sub(r'(\d+)', '{}', name, count=1)
            if abstract_name in weight_map:
                layer_num = re.search(r'\d+', name).group(0)
                new_state_dict[weight_map[abstract_name].format(layer_num)] = param

    for name, param in new_state_dict.items():
        print(f'{name} {param.dtype} {param.shape}')

    os.makedirs(save_path, exist_ok=True)
    torch.save(new_state_dict, save_path / 'model.pth')
    shutil.copy(hf_repo_path / 'tokenizer.json', save_path / 'tokenizer.json')
    if os.path.exists(hf_repo_path / 'special_tokens_map.json'):
        shutil.copy(hf_repo_path / 'special_tokens_map.json', save_path / 'special_tokens_map.json')
    if os.path.exists(hf_repo_path / 'tokenizer_config.json'):
        shutil.copy(hf_repo_path / 'tokenizer_config.json', save_path / 'tokenizer_config.json')
    if os.path.exists(hf_repo_path / 'vocab.json'):
        shutil.copy(hf_repo_path / 'vocab.json', save_path / 'vocab.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hf_repo_path', type=str, default='./checkpoints/llama-7b')
    parser.add_argument('--save_path', type=str, default='./checkpoints/DeepSeek-R1-Distill-Qwen-1.5B/convert')
    parser.add_argument('--weight_map_path', type=str, default='./checkpoints/weight_map/qwen2.5.json')

    args = parser.parse_args()

    convert(Path(args.hf_repo_path), Path(args.save_path), Path(args.weight_map_path))
