import time
import torch
from models.utils import dynamically_quantize_per_channel, _check_linear_int4_k, find_multiple, \
    prepare_int4_weight_and_scales_and_zeros
import argparse
from pathlib import Path
from models.utils import Config
import torch.nn.functional as F


def is_linear(name, linear_names):
    for linear_name in linear_names:
        if linear_name in name:
            return True
    return False


def main(checkpoint_path: Path, quantize: str, groupsize: int = 128):
    linear_names = ['linear']

    weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    quantized_state_dict = {}
    if quantize == 'int8':
        for name, param in weights.items():
            if is_linear(name, linear_names) and name.endswith('weight'):
                int8_weight, scales, _ = dynamically_quantize_per_channel(param.float(), -128, 127, torch.int8)
                quantized_state_dict[name] = int8_weight
                quantized_state_dict[name.replace('weight', 'scales')] = scales.to(param.dtype)
                del int8_weight, scales, _
            else:
                quantized_state_dict[name] = param
            weights[name] = None

        new_pth_name = checkpoint_path.name.replace('.pth', '_int8.pth')
    elif quantize == 'int4':
        inner_k_tiles = 8
        for name, param in weights.items():
            if is_linear(name, linear_names):
                out_features, in_features = param.shape
                assert out_features % 8 == 0, 'require out_features % 8 == 0'

                weight = param.data
                if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
                    padded_in_features = find_multiple(in_features, 1024)
                    weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                else:
                    continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(param.dtype), groupsize, inner_k_tiles
                )
                quantized_state_dict[name] = weight_int4pack
                quantized_state_dict[name.replace('weight', 'scales_and_zeros')] = scales_and_zeros
            else:
                quantized_state_dict[name] = param
            weights[name] = None

        new_pth_name = checkpoint_path.name.replace('.pth', f'_int4.g{groupsize}.pth')
    else:
        raise

    for name, param in quantized_state_dict.items():
        print(f'{name} {param.dtype} {param.shape}')

    torch.save(quantized_state_dict, checkpoint_path.parent / new_pth_name)
    print(f'Quantized weights saved to {checkpoint_path.parent / new_pth_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/pythia-2.8b/convert/model.pth')
    parser.add_argument('--quantize', type=str, default='int8', choices=['int8', 'int4'])

    # for int4 quantization
    parser.add_argument('--groupsize', type=int, default=32)

    args = parser.parse_args()

    main(Path(args.checkpoint_path), args.quantize, groupsize=args.groupsize)
