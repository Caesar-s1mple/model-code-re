import argparse
import torch
from pathlib import Path
from runs import ar, ss, msts_generation
from models.utils import set_seed

default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
set_seed(123456)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 常规设置
    parser.add_argument('--prompt', type=str, default="你是谁？")
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--use_cache', type=bool, default=True)
    # 模型指定
    parser.add_argument('--config_path', type=str, default='./models/config/qwen2.5-1.5b.json')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/DeepSeek-R1-Distill-Qwen-1.5B/convert/model.pth')
    parser.add_argument('--quantize', type=str, default=None, choices=[None, 'int8', 'int4'])
    parser.add_argument('--dialogue', type=bool, default=True)
    parser.add_argument('--system_prompt', type=str, default='')
    # 采样策略
    parser.add_argument('--sample_strategy', type=str, default='ar', choices=['ar', 'ss', 'msts_generation'])
    # 如果使用ss，则需设置以下draft model
    parser.add_argument('--draft_config_path', type=str, default='./models/config/llama-2-160m.json')
    parser.add_argument('--draft_checkpoint_path', type=str, default='./checkpoints/llama-160m/convert/model.pth')
    parser.add_argument('--draft_quantize', type=str, default=None, choices=[None, 'int8', 'int4'])
    parser.add_argument('--gamma', type=int, default=7, help='draft tokens个数')

    args = parser.parse_args()

    if args.sample_strategy == 'ar':
        ar.main(prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                config_path=Path(args.config_path),
                checkpoint_path=Path(args.checkpoint_path),
                quantize=args.quantize,
                use_cache=args.use_cache,
                dialogue=args.dialogue,
                system_prompt=args.system_prompt
                )
    elif args.sample_strategy == 'ss':
        ss.main(prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                config_path=Path(args.config_path),
                checkpoint_path=Path(args.checkpoint_path),
                draft_config_path=Path(args.draft_config_path),
                draft_checkpoint_path=Path(args.draft_checkpoint_path),
                quantize=args.quantize,
                draft_quantize=args.draft_quantize,
                gamma=args.gamma,
                use_cache=args.use_cache,
                dialogue=args.dialogue,
                system_prompt=args.system_prompt
                )
    elif args.sample_strategy == 'msts_generation':
        msts_generation.main(max_new_tokens=args.max_new_tokens,
                             config_path=Path(args.config_path),
                             checkpoint_path=Path(args.checkpoint_path),
                             quantize=args.quantize,
                             use_cache=args.use_cache,
                             device=args.device,
                             temperature=args.temperature,
                             top_k=args.top_k,
                             top_p=args.top_p
                             )
