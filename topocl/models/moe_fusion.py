#!/usr/bin/env python3
"""5-Expert Mixture of Experts for Visual-Topology Fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFusedEncoder(nn.Module):
    """
    5-Expert Mixture of Experts for Visual-Topology Fusion.
    Processes ONE view at a time (img, h0, h1).

    Reads topo_encoder.hidden_dim * 4 dynamically for topo_proj input dim.
    """

    def __init__(self, visual_encoder, topo_encoder, embed_dim=256, out_dim=512, dropout=0.2):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.topo_encoder = topo_encoder

        vis_dim = visual_encoder.out_dim  # 2048
        topo_dim = topo_encoder.hidden_dim * 4  # Dynamic: hidden_dim * 4

        # Common projections
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.topo_proj = nn.Linear(topo_dim, embed_dim)

        # Expert 1: Visual-only
        self.expert_vis = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Expert 2: Topology-only
        self.expert_topo = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Expert 3: Concatenation
        self.expert_concat = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Expert 4: Cross-Attention (2 heads)
        self.cross_attn_vis = nn.MultiheadAttention(
            embed_dim, num_heads=2, dropout=dropout, batch_first=True
        )
        self.cross_attn_topo = nn.MultiheadAttention(
            embed_dim, num_heads=2, dropout=dropout, batch_first=True
        )
        self.norm_vis = nn.LayerNorm(embed_dim)
        self.norm_topo = nn.LayerNorm(embed_dim)
        self.expert_attention = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Expert 5: Gated Fusion
        self.gate_network = nn.Sequential(
            nn.Linear(vis_dim + topo_dim, embed_dim),
            nn.Sigmoid()
        )
        self.expert_gated = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Meta-gating
        self.meta_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5)
        )

        self.out_dim = out_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img, h0, h1):
        """
        Process ONE view (image + topology features).

        Args:
            img: (B, 3, H, W) augmented image
            h0: (B, n_h0, 2) H0 persistence features
            h1: (B, n_h1, 2) H1 persistence features

        Returns:
            fused: (B, out_dim) fused features
            gates: (B, 5) expert gating weights
        """
        # Encode visual and topology features
        vis_feat = self.visual_encoder(img)  # (B, 2048)
        topo_feat = self.topo_encoder(h0, h1)  # (B, topo_dim)

        # Project to common embedding space
        vis_embed = self.vis_proj(vis_feat)  # (B, embed_dim)
        topo_embed = self.topo_proj(topo_feat)  # (B, embed_dim)

        # Expert 1: Visual-only
        expert1_out = self.expert_vis(vis_embed)

        # Expert 2: Topology-only
        expert2_out = self.expert_topo(topo_embed)

        # Expert 3: Concatenation
        concat_feat = torch.cat([vis_embed, topo_embed], dim=1)
        expert3_out = self.expert_concat(concat_feat)

        # Expert 4: Cross-Attention
        vis_seq = vis_embed.unsqueeze(1)
        topo_seq = topo_embed.unsqueeze(1)

        vis_refined, _ = self.cross_attn_vis(query=vis_seq, key=topo_seq, value=topo_seq)
        topo_refined, _ = self.cross_attn_topo(query=topo_seq, key=vis_seq, value=vis_seq)

        vis_out = self.norm_vis(vis_seq + vis_refined).squeeze(1)
        topo_out = self.norm_topo(topo_seq + topo_refined).squeeze(1)

        attn_feat = torch.cat([vis_out, topo_out], dim=1)
        expert4_out = self.expert_attention(attn_feat)

        # Expert 5: Gated Fusion
        gate_z = self.gate_network(torch.cat([vis_feat, topo_feat], dim=1))
        gated_feat = gate_z * vis_embed + (1 - gate_z) * topo_embed
        expert5_out = self.expert_gated(gated_feat)

        # Meta-gating
        meta_gate_input = torch.cat([vis_embed, topo_embed], dim=1)
        meta_gate_logits = self.meta_gate(meta_gate_input)
        gates = F.softmax(meta_gate_logits / 2.0, dim=1)

        # Weighted combination
        fused = (gates[:, 0:1] * expert1_out +
                 gates[:, 1:2] * expert2_out +
                 gates[:, 2:3] * expert3_out +
                 gates[:, 3:4] * expert4_out +
                 gates[:, 4:5] * expert5_out)

        return fused, gates
