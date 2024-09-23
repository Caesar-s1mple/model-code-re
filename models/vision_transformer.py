import torch
import torch.nn as nn
import math

import torchvision


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size

        pass


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()


