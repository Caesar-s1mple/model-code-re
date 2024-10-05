from .llama_compile import LLaMA as LLaMACompile
from .llama import LLaMA
from .transformer import Transformer
from .bert import Bert
from .gpt_neox import GPTNeoX

model_map = {
    'LLaMA': LLaMA,
    'LLaMACompile': LLaMACompile,
    'Transformer': Transformer,
    'Bert': Bert,
    'GPTNeoX': GPTNeoX
}

__all__ = [
    'LLaMACompile',
    'LLaMA',
    'Transformer',
    'Bert',
    'GPTNeoX',
    'model_map'
]
