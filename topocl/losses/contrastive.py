#!/usr/bin/env python3
"""Contrastive loss functions"""

import torch
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.1):
    """InfoNCE loss for contrastive learning"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # Mask out self-similarities
    mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_indices = torch.cat([
        torch.arange(N, 2 * N, device=sim.device),
        torch.arange(0, N, device=sim.device)
    ])

    loss = F.cross_entropy(sim, pos_indices)
    return loss


def supcon_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss"""
    features = F.normalize(features, dim=1)

    batch_size = features.shape[0]

    # Compute similarity matrix
    sim_matrix = torch.mm(features, features.t()) / temperature

    # Mask out self-similarities
    mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, float('-inf'))

    # Create label mask
    labels = labels.contiguous().view(-1, 1)
    label_mask = torch.eq(labels, labels.t()).float()
    label_mask.masked_fill_(mask, 0)

    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()

    return loss
