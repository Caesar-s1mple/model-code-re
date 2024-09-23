import argparse
import torch
import importlib

default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', type=str, default='Hello, my name is')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--top_p', type=float, default=0.)
    parser.add_argument('--device', type=str, default=default_device)

    parser.add_argument('--config_path', type=str, default='./models/config/llama-3-8b_compile.json')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/Meta-Llama-3-8B/convert/model.pth')
    parser.add_argument('--quantize', type=str, default=None, choices=['int8'])

    parser.add_argument('--sample_strategy', type=str, default='ar', choices=['ar', 'ss'])
    parser.add_argument('--model', type=str, default='llama_compile')

    args = parser.parse_args()

    if args.sample_strategy == 'ar':
        module_name = f'runs_ar.{args.model}'
        module = importlib.import_module(module_name)
        module.main(prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                    config_path=args.config_path,
                    checkpoint_path=args.checkpoint_path,
                    quantize=args.quantize
                    )
    elif args.sample_strategy == 'ss':
        module_name = f'from runs_ss.{args.model}'
        module = importlib.import_module(module_name)
        module.main(prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                    config_path=args.config_path,
                    checkpoint_path=args.checkpoint_path,
                    draft_config_path=args.draft_config_path,
                    draft_checkpoint_path=args.draft_checkpoint_path,
                    quantize=args.quantize
                    )
