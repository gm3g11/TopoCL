#!/usr/bin/env python3
"""Hierarchical Topology Encoder for persistent homology features"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalTopoEncoder(nn.Module):
    """
    Hierarchical encoder for persistent homology features.

    Separate processing for H0 and H1 with cross-homology attention.
    Paper defaults: hidden_dim=384, num_heads=4, n_h0=48, n_h1=96.
    """

    def __init__(
            self,
            hidden_dim: int = 384,
            num_heads: int = 4,
            dropout: float = 0.15,
            n_h0: int = 48,
            n_h1: int = 96
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_h0 = n_h0
        self.n_h1 = n_h1

        # Separate encoders for H0 and H1
        self.h0_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )

        self.h1_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )

        # Self-attention for H0 and H1
        self.h0_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.h1_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention between H0 and H1
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output dimension: max+mean for H0 and H1
        self.out_dim = hidden_dim * 4

        print(f"  TopoEncoder: H0={n_h0}, H1={n_h1}, hidden={hidden_dim}, heads={num_heads}, dropout={dropout}")

    def forward(self, h0, h1):
        """
        Args:
            h0: (B, n_h0, 2) H0 birth-death pairs
            h1: (B, n_h1, 2) H1 birth-death pairs

        Returns:
            features: (B, out_dim)
        """
        # Encode H0 and H1 separately
        h0_feat = self.h0_encoder(h0)  # (B, n_h0, hidden_dim)
        h1_feat = self.h1_encoder(h1)  # (B, n_h1, hidden_dim)

        # Self-attention within each homology dimension
        h0_attn, _ = self.h0_attention(h0_feat, h0_feat, h0_feat)
        h0_feat = h0_feat + h0_attn

        h1_attn, _ = self.h1_attention(h1_feat, h1_feat, h1_feat)
        h1_feat = h1_feat + h1_attn

        # Cross-attention: H0 attends to H1
        h0_cross, _ = self.cross_attention(h0_feat, h1_feat, h1_feat)

        # Global pooling
        h0_max = h0_feat.max(dim=1).values
        h0_mean = h0_feat.mean(dim=1)
        h1_max = h1_feat.max(dim=1).values
        h1_mean = h1_feat.mean(dim=1)

        # Concatenate all features
        features = torch.cat([h0_max, h0_mean, h1_max, h1_mean], dim=1)

        return features
