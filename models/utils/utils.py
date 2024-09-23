import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random
import numpy as np


class Config:
    def __init__(self, config_path):
        with open(os.path.join(config_path), 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def __getattr__(self, item):
        return self.config[item]


def sample(probs: torch.Tensor, num_samples: int = 1):
    idx = torch.multinomial(probs, num_samples=num_samples)

    return idx


def norm_logits(logits: torch.Tensor, temperature: float = 1, top_k: int = 0, top_p: float = 0.):
    logits = logits / temperature
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = -torch.inf

    if top_p > 0.:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = -torch.inf

    probs = F.softmax(logits, dim=-1)

    return probs


def norm_max(px, qx):
    """
    norm(max(0, px - qx))
    used in speculative sampling
    """
    x = px - qx
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
