import torch
from models import bert, gpt2, transformer, vision_transformer
from models.config.config import Config

if __name__ == '__main__':
    config = Config('transformer')
    model = transformer.Transformer(config)
