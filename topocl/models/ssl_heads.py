#!/usr/bin/env python3
"""Stage 3 SSL Model Wrappers (wrap MoEFusedEncoder for contrastive learning)"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(in_dim, hidden_dim, out_dim, dropout=0.0):
    """3-layer MLP with BatchNorm"""
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
    return nn.Sequential(*layers)


class SimCLRModel(nn.Module):
    """SimCLR with MoE fusion"""

    def __init__(self, moe_encoder, proj_dim=256):
        super().__init__()
        self.encoder = moe_encoder
        self.projector = MLP(moe_encoder.out_dim, 512, proj_dim)

    def forward(self, img1, img2, h0_1, h1_1, h0_2, h1_2, temperature=0.2):
        z1, gates1 = self.encoder(img1, h0_1, h1_1)
        z1 = F.normalize(self.projector(z1), dim=1)

        z2, gates2 = self.encoder(img2, h0_2, h1_2)
        z2 = F.normalize(self.projector(z2), dim=1)

        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / temperature

        mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=sim.device),
            torch.arange(0, N, device=sim.device)
        ])

        loss = F.cross_entropy(sim, pos_indices)
        gates = (gates1 + gates2) / 2.0
        return loss, gates


class BYOLModel(nn.Module):
    """BYOL with MoE fusion"""

    def __init__(self, moe_encoder, proj_dim=256, m=0.99):
        super().__init__()
        self.m = m

        self.online_encoder = moe_encoder
        self.online_projector = MLP(moe_encoder.out_dim, 512, proj_dim)
        self.predictor = MLP(proj_dim, 512, proj_dim)

        from copy import deepcopy
        self.target_encoder = deepcopy(moe_encoder)
        self.target_projector = MLP(moe_encoder.out_dim, 512, proj_dim)

        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False
        for p_o, p_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def _update_target(self):
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data = p_t.data * self.m + p_o.data * (1. - self.m)
        for p_o, p_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            p_t.data = p_t.data * self.m + p_o.data * (1. - self.m)

    def forward(self, img1, img2, h0_1, h1_1, h0_2, h1_2, **kwargs):
        feat1, gates1 = self.online_encoder(img1, h0_1, h1_1)
        p1 = self.predictor(self.online_projector(feat1))

        feat2, gates2 = self.online_encoder(img2, h0_2, h1_2)
        p2 = self.predictor(self.online_projector(feat2))

        with torch.no_grad():
            self._update_target()
            z1, _ = self.target_encoder(img1, h0_1, h1_1)
            z1 = self.target_projector(z1)
            z2, _ = self.target_encoder(img2, h0_2, h1_2)
            z2 = self.target_projector(z2)

        def loss_fn(p, z):
            return 2 - 2 * (F.normalize(p, dim=1) * F.normalize(z, dim=1)).sum(dim=1)

        loss = (loss_fn(p1, z2).mean() + loss_fn(p2, z1).mean()) / 2
        gates = (gates1 + gates2) / 2.0
        return loss, gates


class DINOModel(nn.Module):
    """DINO with MoE fusion"""

    def __init__(self, moe_encoder, proj_dim=256, student_temp=0.1, teacher_temp=0.04, m=0.996):
        super().__init__()
        self.m = m
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        self.student_encoder = moe_encoder
        self.student_projector = nn.Sequential(
            nn.Linear(moe_encoder.out_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

        from copy import deepcopy
        self.teacher_encoder = deepcopy(moe_encoder)
        self.teacher_projector = nn.Sequential(
            nn.Linear(moe_encoder.out_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )

        self.register_buffer("center", torch.zeros(1, proj_dim))
        self._init_teacher()

    @torch.no_grad()
    def _init_teacher(self):
        for p_s, p_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False
        for p_s, p_t in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self):
        for p_s, p_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            p_t.data = p_t.data * self.m + p_s.data * (1. - self.m)
        for p_s, p_t in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            p_t.data = p_t.data * self.m + p_s.data * (1. - self.m)

    def forward(self, img1, img2, h0_1, h1_1, h0_2, h1_2, **kwargs):
        feat1, gates1 = self.student_encoder(img1, h0_1, h1_1)
        s1 = F.normalize(self.student_projector(feat1), dim=-1, p=2)

        feat2, gates2 = self.student_encoder(img2, h0_2, h1_2)
        s2 = F.normalize(self.student_projector(feat2), dim=-1, p=2)

        with torch.no_grad():
            self._update_teacher()

            t_feat1, _ = self.teacher_encoder(img1, h0_1, h1_1)
            t1 = F.normalize(self.teacher_projector(t_feat1), dim=-1, p=2)

            t_feat2, _ = self.teacher_encoder(img2, h0_2, h1_2)
            t2 = F.normalize(self.teacher_projector(t_feat2), dim=-1, p=2)

            batch_center = torch.cat([t1, t2], dim=0).mean(dim=0, keepdim=True)
            self.center = self.center * 0.9 + batch_center * 0.1

        t1_centered = t1 - self.center
        t2_centered = t2 - self.center

        loss1 = -torch.sum(
            F.softmax(t2_centered / self.teacher_temp, dim=-1) *
            F.log_softmax(s1 / self.student_temp, dim=-1),
            dim=-1
        ).mean()

        loss2 = -torch.sum(
            F.softmax(t1_centered / self.teacher_temp, dim=-1) *
            F.log_softmax(s2 / self.student_temp, dim=-1),
            dim=-1
        ).mean()

        loss = (loss1 + loss2) / 2
        gates = (gates1 + gates2) / 2.0
        return loss, gates


class BarlowModel(nn.Module):
    """Barlow Twins with MoE fusion"""

    def __init__(self, moe_encoder, proj_dim=128, lambd=0.005):
        super().__init__()
        self.encoder = moe_encoder
        self.lambd = lambd

        self.projector = nn.Sequential(
            nn.Linear(moe_encoder.out_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim, bias=False)
        )
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, img1, img2, h0_1, h1_1, h0_2, h1_2, **kwargs):
        feat1, gates1 = self.encoder(img1, h0_1, h1_1)
        z1 = self.bn(self.projector(feat1))

        feat2, gates2 = self.encoder(img2, h0_2, h1_2)
        z2 = self.bn(self.projector(feat2))

        # Cross-correlation matrix
        c = (z1.T @ z2) / z1.size(0)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(c.size(0) - 1, c.size(0) + 1)[:, :-1].pow_(2).sum()

        loss = on_diag + self.lambd * off_diag
        gates = (gates1 + gates2) / 2.0
        return loss, gates


class MoCoV3Model(nn.Module):
    """MoCo v3 with MoE fusion"""

    def __init__(self, moe_encoder, proj_dim=256, m=0.996):
        super().__init__()
        self.m = m

        self.encoder_q = moe_encoder
        self.projector_q = MLP(moe_encoder.out_dim, 2048, proj_dim, dropout=0.1)
        self.predictor = MLP(proj_dim, 2048, proj_dim, dropout=0.1)

        from copy import deepcopy
        self.encoder_k = deepcopy(moe_encoder)
        self.projector_k = MLP(moe_encoder.out_dim, 2048, proj_dim)

        self._init_key()

    @torch.no_grad()
    def _init_key(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        for p_q, p_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    @torch.no_grad()
    def _update_key(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)
        for p_q, p_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    def forward(self, img1, img2, h0_1, h1_1, h0_2, h1_2, temperature=0.2):
        feat_q1, gates1 = self.encoder_q(img1, h0_1, h1_1)
        q1 = self.predictor(self.projector_q(feat_q1))

        feat_q2, gates2 = self.encoder_q(img2, h0_2, h1_2)
        q2 = self.predictor(self.projector_q(feat_q2))

        with torch.no_grad():
            self._update_key()
            k1, _ = self.encoder_k(img1, h0_1, h1_1)
            k1 = self.projector_k(k1)
            k2, _ = self.encoder_k(img2, h0_2, h1_2)
            k2 = self.projector_k(k2)

        def contrastive_loss(q, k):
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)
            logits = torch.mm(q, k.t()) / temperature
            labels = torch.arange(logits.shape[0], device=q.device)
            return F.cross_entropy(logits, labels)

        loss = contrastive_loss(q1, k2) + contrastive_loss(q2, k1)
        gates = (gates1 + gates2) / 2.0
        return loss, gates
