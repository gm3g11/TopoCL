#!/usr/bin/env python3
"""Training utilities"""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset: str) -> int:
    """Get number of classes for each dataset"""
    class_map = {
        'kvasir': 8,
        'pathmnist': 9,
        'octmnist': 4,
        'organsmnist': 11,
        'organamnist': 11,
        'isic2019': 8,
    }
    return class_map.get(dataset.lower(), 10)


def balance_loss(gate_weights, low_threshold=0.05, high_threshold=0.5):
    """Balance loss to prevent expert collapse"""
    importance = gate_weights.mean(dim=0) + 1e-8
    dominance_penalty = F.relu(importance - high_threshold).sum()
    ignore_penalty = F.relu(low_threshold - importance).sum()
    return torch.clamp(dominance_penalty + ignore_penalty, 0, 5)


def print_expert_statistics(epoch, gate_weights, method_name):
    """Print expert gating statistics"""
    importance = gate_weights.mean(dim=0).cpu().numpy()
    std = gate_weights.std(dim=0).cpu().numpy()
    max_weight = gate_weights.max(dim=0)[0].cpu().numpy()
    min_weight = gate_weights.min(dim=0)[0].cpu().numpy()

    entropy = -(importance * np.log(importance + 1e-8)).sum()
    max_entropy = np.log(5)

    print(f"\n{'=' * 80}")
    print(f"Expert Statistics - {method_name.upper()} - Epoch {epoch}")
    print(f"{'=' * 80}")
    print(f"{'Expert':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-' * 80}")

    expert_names = ["Visual-only", "Topology-only", "Concatenation",
                    "Cross-Attention", "Gated Fusion"]

    for i, name in enumerate(expert_names):
        print(f"{name:<20} {importance[i]:<10.4f} {std[i]:<10.4f} "
              f"{min_weight[i]:<10.4f} {max_weight[i]:<10.4f}")

    print(f"{'-' * 80}")
    print(f"Gating Entropy: {entropy:.4f} / {max_entropy:.4f} "
          f"({entropy / max_entropy * 100:.1f}%)")

    dominant = importance > 0.5
    ignored = importance < 0.05

    if dominant.any():
        print(f"Dominant experts (>0.5): {[expert_names[i] for i in range(5) if dominant[i]]}")
    if ignored.any():
        print(f"Ignored experts (<0.05): {[expert_names[i] for i in range(5) if ignored[i]]}")
    if not dominant.any() and not ignored.any():
        print(f"All experts in healthy range [0.05, 0.5]")

    print(f"{'=' * 80}\n")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
