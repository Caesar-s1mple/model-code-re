import math

import torch
import torch.nn as nn
from typing import Optional, List
from torch import Tensor


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dropout=0.):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank

        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: Tensor):
        return self.B(self.dropout(self.A(x)) * self.scaling)


class LinearWithLoRA(nn.Module):
    def __init(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.):
        super().__init()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dropout)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    def merge(self):
        A_weight = self.lora.A.weight
        B_weight = self.lora.B.weight
        scaling = self.lora.scaling
        lora_weight = torch.matmul(A_weight, B_weight) * scaling

        self.linear.weight.data += lora_weight

        self.lora.A = None
        self.lora.B = None
        self.lora.scaling = None


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float, dropout: float = 0., except_modules: Optional[List[str]] = None):
    if except_modules is None:
        except_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name not in except_modules:
            model._modules[name] = LinearWithLoRA(module, rank, alpha, dropout)
