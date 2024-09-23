import os
import torch
from pathlib import Path
import re
import shutil

weight_map = {
    'tok_embeddings.weight': 'word_embeddings.weight',
    'layers.{}.attention.wq.weight': 'layers.{}.self_attn.linear_q.weight',
    'layers.{}.attention.wk.weight': 'layers.{}.self_attn.linear_k.weight',
    'layers.{}.attention.wv.weight': 'layers.{}.self_attn.linear_v.weight',
    'layers.{}.attention.wo.weight': 'layers.{}.self_attn.fc.weight',
    'layers.{}.feed_forward.w1.weight': 'layers.{}.ff.linear1.0.weight',
    'layers.{}.feed_forward.w3.weight': 'layers.{}.ff.linear2.weight',
    'layers.{}.feed_forward.w2.weight': 'layers.{}.ff.linear3.weight',
    'layers.{}.attention_norm.weight': 'layers.{}.attn_norm.weight',
    'layers.{}.ffn_norm.weight': 'layers.{}.ff_norm.weight',
    'norm.weight': 'norm.weight',
    'output.weight': 'linear.weight'
}


def convert(hf_repo_path: Path):
    original_weights_path = hf_repo_path / 'original' / 'consolidated.00.pth'
    original_weights = torch.load(original_weights_path, map_location='cpu', weights_only=True)

    new_state_dict = {}
    for name, param in original_weights.items():
        if name in weight_map:
            new_name = weight_map[name]
            new_state_dict[new_name] = param
        else:
            abstract_name = re.sub(r'(\d+)', '{}', name, count=1)
            layer_num = re.search(r'\d+', name).group(0)
            new_state_dict[weight_map[abstract_name].format(layer_num)] = param

    os.makedirs(hf_repo_path / 'convert', exist_ok=True)
    torch.save(new_state_dict, hf_repo_path / 'convert' / 'model.pth')
    shutil.copy(hf_repo_path / 'original' / 'tokenizer.model', hf_repo_path / 'convert' / 'tokenizer.model')


if __name__ == '__main__':
    hf_repo_path = './checkpoints/Meta-Llama-3-8B'
    convert(Path(hf_repo_path))
