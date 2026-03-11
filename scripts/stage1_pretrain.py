#!/usr/bin/env python3
"""
Stage 1: Unified Visual Encoder Pretraining

Consolidates SimCLR, MoCo-v3, BYOL, Barlow Twins, and DINO into a single script.

Usage:
  python stage1_pretrain.py --method simclr --dataset pathmnist
  python stage1_pretrain.py --method dino --dataset isic2019 --data_dir ~/afs/isic2019
  python stage1_pretrain.py --method byol --dataset kvasir --epochs 200 --batch_size 128
"""

import os
import math
import json
import random
import argparse
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from copy import deepcopy

from topocl.data.augmentations import (
    SimCLRAugmentation,
    MoCoAugmentation,
    BarlowAugmentation,
    DINOAugmentation,
    EvalTransform,
    EnsureRGB,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODER_OUT_DIM = 2048

METHODS = ["simclr", "mocov3", "byol", "barlow", "dino"]
DATASETS = ["pathmnist", "octmnist", "organsmnist", "isic2019", "kvasir"]

NUM_CLASSES = {
    "pathmnist": 9,
    "octmnist": 4,
    "organsmnist": 11,
    "isic2019": 8,
    "kvasir": 8,
}

# Per-method defaults  -------------------------------------------------------
METHOD_DEFAULTS = {
    "simclr": dict(
        batch_size=512, epochs=200, lr=3e-3, weight_decay=1e-4,
        temperature=0.5, proj_hidden=2048, proj_out=128, momentum_coeff=None,
    ),
    "mocov3": dict(
        batch_size=256, epochs=100, lr=3e-4, weight_decay=0.1,
        temperature=0.2, proj_hidden=4096, proj_out=256, momentum_coeff=0.99,
    ),
    "byol": dict(
        batch_size=256, epochs=100, lr=3e-4, weight_decay=1.5e-6,
        temperature=None, proj_hidden=4096, proj_out=256, momentum_coeff=0.99,
    ),
    "barlow": dict(
        batch_size=256, epochs=100, lr=1e-3, weight_decay=1e-6,
        temperature=None, proj_hidden=2048, proj_out=512, momentum_coeff=None,
    ),
    "dino": dict(
        batch_size=256, epochs=100, lr=5e-4, weight_decay=0.04,
        temperature=None, proj_hidden=2048, proj_out=65536, momentum_coeff=0.996,
    ),
}

SMALL_DATASETS = {"pathmnist", "octmnist", "organsmnist"}


# ---------------------------------------------------------------------------
# Backbone helper
# ---------------------------------------------------------------------------
def make_resnet50_encoder():
    """Create a ResNet-50 encoder with identity FC, outputting 2048-d."""
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    return backbone


# ============================================================================
# SSL Wrapper Classes
# ============================================================================

class SimCLRWrapper(nn.Module):
    """SimCLR: ResNet50 + 2-layer projector, NT-Xent loss."""

    def __init__(self, proj_hidden=2048, proj_out=128, temperature=0.5):
        super().__init__()
        self.encoder = make_resnet50_encoder()
        self.projector = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )
        self.temperature = temperature

    def forward(self, x1, x2):
        z1 = F.normalize(self.projector(self.encoder(x1)), dim=1)
        z2 = F.normalize(self.projector(self.encoder(x2)), dim=1)

        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=sim.device),
            torch.arange(0, N, device=sim.device),
        ])
        loss = F.cross_entropy(sim, pos_indices)
        return loss

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()


class MoCoV3Wrapper(nn.Module):
    """MoCo-v3: query + key encoder (EMA), 2-layer projector + predictor."""

    def __init__(self, proj_hidden=4096, proj_out=256, temperature=0.2,
                 momentum_coeff=0.99):
        super().__init__()
        self.temperature = temperature
        self.m = momentum_coeff

        # Query path
        self.encoder_q = make_resnet50_encoder()
        self.projector_q = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_out, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )

        # Key path (EMA)
        self.encoder_k = deepcopy(self.encoder_q)
        self.projector_k = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )
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
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)
        for p_q, p_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)

    def _contrastive_loss(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        logits = torch.mm(q, k.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(self, x1, x2):
        q1 = self.predictor(self.projector_q(self.encoder_q(x1)))
        q2 = self.predictor(self.projector_q(self.encoder_q(x2)))

        with torch.no_grad():
            self._momentum_update()
            k1 = self.projector_k(self.encoder_k(x1))
            k2 = self.projector_k(self.encoder_k(x2))

        loss = self._contrastive_loss(q1, k2) + self._contrastive_loss(q2, k1)
        return loss

    def get_encoder_state_dict(self):
        return self.encoder_q.state_dict()


class BYOLWrapper(nn.Module):
    """BYOL: online + target encoder (EMA), predictor on online branch, cosine loss."""

    def __init__(self, proj_hidden=4096, proj_out=256, momentum_coeff=0.99):
        super().__init__()
        self.m = momentum_coeff

        # Online path
        self.online_encoder = make_resnet50_encoder()
        self.online_projector = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_out, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )

        # Target path (EMA)
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )
        self._init_target()

    @torch.no_grad()
    def _init_target(self):
        for p_o, p_t in zip(self.online_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False
        for p_o, p_t in zip(self.online_projector.parameters(),
                            self.target_projector.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for p_o, p_t in zip(self.online_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data = p_t.data * self.m + p_o.data * (1.0 - self.m)
        for p_o, p_t in zip(self.online_projector.parameters(),
                            self.target_projector.parameters()):
            p_t.data = p_t.data * self.m + p_o.data * (1.0 - self.m)

    @staticmethod
    def _cosine_loss(p, z):
        return 2.0 - 2.0 * (F.normalize(p, dim=1) * F.normalize(z, dim=1)).sum(dim=1)

    def forward(self, x1, x2):
        # Online
        p1 = self.predictor(self.online_projector(self.online_encoder(x1)))
        p2 = self.predictor(self.online_projector(self.online_encoder(x2)))

        with torch.no_grad():
            self._momentum_update()
            z1 = self.target_projector(self.target_encoder(x1))
            z2 = self.target_projector(self.target_encoder(x2))

        loss = (self._cosine_loss(p1, z2).mean() + self._cosine_loss(p2, z1).mean()) / 2.0
        return loss

    def get_encoder_state_dict(self):
        return self.online_encoder.state_dict()


class BarlowTwinsWrapper(nn.Module):
    """Barlow Twins: 3-layer bias-free projector with BN, cross-correlation loss."""

    def __init__(self, proj_hidden=2048, proj_out=512, lambd=0.005):
        super().__init__()
        self.lambd = lambd

        self.encoder = make_resnet50_encoder()
        # 3-layer bias-free projector with BN
        self.projector = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out, bias=False),
        )
        self.bn = nn.BatchNorm1d(proj_out, affine=False)

    def forward(self, x1, x2):
        z1 = self.bn(self.projector(self.encoder(x1)))
        z2 = self.bn(self.projector(self.encoder(x2)))

        # Cross-correlation matrix
        c = (z1.T @ z2) / z1.size(0)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c.flatten()[1:]
                    .view(c.size(0) - 1, c.size(0) + 1)[:, :-1]
                    .pow_(2).sum())

        loss = on_diag + self.lambd * off_diag
        return loss

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()


class DINOWrapper(nn.Module):
    """DINO: student + teacher with centering, 3-layer DINO head with GELU + bottleneck."""

    def __init__(self, proj_hidden=2048, proj_out=65536,
                 student_temp=0.1, teacher_temp=0.04, momentum_coeff=0.996):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.m = momentum_coeff

        # Student
        self.student_encoder = make_resnet50_encoder()
        self.student_head = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_out),
        )

        # Teacher (EMA)
        self.teacher_encoder = deepcopy(self.student_encoder)
        self.teacher_head = nn.Sequential(
            nn.Linear(ENCODER_OUT_DIM, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_out),
        )

        self.register_buffer("center", torch.zeros(1, proj_out))
        self._init_teacher()

    @torch.no_grad()
    def _init_teacher(self):
        for p_s, p_t in zip(self.student_encoder.parameters(),
                            self.teacher_encoder.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False
        for p_s, p_t in zip(self.student_head.parameters(),
                            self.teacher_head.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for p_s, p_t in zip(self.student_encoder.parameters(),
                            self.teacher_encoder.parameters()):
            p_t.data = p_t.data * self.m + p_s.data * (1.0 - self.m)
        for p_s, p_t in zip(self.student_head.parameters(),
                            self.teacher_head.parameters()):
            p_t.data = p_t.data * self.m + p_s.data * (1.0 - self.m)

    def forward(self, crops):
        """
        Args:
            crops: list of tensors. First 2 are global crops, rest are local crops.
                   All on the same device.
        Returns:
            scalar loss
        """
        # Student: all crops
        student_out = []
        for crop in crops:
            feat = self.student_encoder(crop)
            logit = self.student_head(feat)
            student_out.append(logit)

        # Teacher: only global crops (first 2)
        with torch.no_grad():
            self._momentum_update()
            teacher_out = []
            for crop in crops[:2]:
                feat = self.teacher_encoder(crop)
                logit = self.teacher_head(feat)
                teacher_out.append(logit)

            # Update center
            batch_center = torch.cat(teacher_out, dim=0).mean(dim=0, keepdim=True)
            self.center = self.center * 0.9 + batch_center * 0.1

        # DINO loss: cross-entropy between teacher (sharpened, centered) and student
        total_loss = 0.0
        n_loss_terms = 0
        for t_idx in range(2):  # teacher global crops
            t_centered = teacher_out[t_idx] - self.center
            t_probs = F.softmax(t_centered / self.teacher_temp, dim=-1)
            for s_idx in range(len(crops)):
                if s_idx == t_idx:
                    continue
                s_log_probs = F.log_softmax(student_out[s_idx] / self.student_temp,
                                            dim=-1)
                total_loss += -torch.sum(t_probs * s_log_probs, dim=-1).mean()
                n_loss_terms += 1

        loss = total_loss / n_loss_terms
        return loss

    def get_encoder_state_dict(self):
        return self.student_encoder.state_dict()


# ============================================================================
# Dataset Utilities
# ============================================================================

class SSLImageDataset(Dataset):
    """Wraps numpy images + PIL-based augmentation for dual-view SSL."""

    def __init__(self, images, transform):
        """
        Args:
            images: (N, H, W, 3) uint8 numpy array
            transform: callable that takes a PIL Image and returns a tuple of
                       tensors (view1, view2)  -- or a list of tensors for DINO.
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        pil_img = Image.fromarray(img)
        views = self.transform(pil_img)
        return views


class SSLImageFolderDataset(Dataset):
    """Lazy-loading version for large ImageFolder datasets (ISIC2019, Kvasir)."""

    def __init__(self, paths, transform, img_size=224):
        self.paths = paths
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        views = self.transform(img)
        return views


def load_medmnist_images(dataset_name, img_size=224):
    """Load MedMNIST train split as numpy uint8 (N, H, W, 3)."""
    import medmnist
    from medmnist import INFO

    dataset_key = None
    for key in INFO.keys():
        if key.lower() == dataset_name.lower():
            dataset_key = key
            break

    if dataset_key is None:
        raise ValueError(f"Dataset {dataset_name} not found in MedMNIST")

    key_lower = dataset_key.lower()
    if key_lower == "octmnist":
        class_name = "OCTMNIST"
    elif key_lower.startswith("organ") and "mnist" in key_lower:
        letter = key_lower.replace("organ", "").replace("mnist", "")
        class_name = f"Organ{letter.upper()}MNIST"
    elif "mnist" in key_lower:
        prefix = key_lower.replace("mnist", "")
        class_name = prefix.capitalize() + "MNIST"
    else:
        class_name = dataset_key

    DataClass = getattr(medmnist, class_name)
    ds = DataClass(split="train", download=True, size=img_size)

    imgs = ds.imgs
    labels = ds.labels.squeeze()

    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8)
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)

    return imgs, labels


def load_medmnist_test_images(dataset_name, img_size=224):
    """Load MedMNIST test split as numpy uint8 (N, H, W, 3)."""
    import medmnist
    from medmnist import INFO

    dataset_key = None
    for key in INFO.keys():
        if key.lower() == dataset_name.lower():
            dataset_key = key
            break

    if dataset_key is None:
        raise ValueError(f"Dataset {dataset_name} not found in MedMNIST")

    key_lower = dataset_key.lower()
    if key_lower == "octmnist":
        class_name = "OCTMNIST"
    elif key_lower.startswith("organ") and "mnist" in key_lower:
        letter = key_lower.replace("organ", "").replace("mnist", "")
        class_name = f"Organ{letter.upper()}MNIST"
    elif "mnist" in key_lower:
        prefix = key_lower.replace("mnist", "")
        class_name = prefix.capitalize() + "MNIST"
    else:
        class_name = dataset_key

    DataClass = getattr(medmnist, class_name)
    ds = DataClass(split="test", download=True, size=img_size)

    imgs = ds.imgs
    labels = ds.labels.squeeze()

    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8)
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)

    return imgs, labels


def load_imagefolder_paths_labels(dataset_name, data_dir, seed=42):
    """Return (train_paths, train_labels, test_paths, test_labels) for ISIC / Kvasir."""
    from sklearn.model_selection import train_test_split

    if dataset_name.lower() == "isic2019":
        if data_dir is None:
            data_dir = os.path.expanduser("~/afs/isic2019")
        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_ds = datasets.ImageFolder(train_dir)
            test_ds = datasets.ImageFolder(test_dir)
            all_paths = [s[0] for s in train_ds.samples] + [s[0] for s in test_ds.samples]
            all_labels = train_ds.targets + test_ds.targets
        else:
            full_ds = datasets.ImageFolder(data_dir)
            all_paths = [s[0] for s in full_ds.samples]
            all_labels = full_ds.targets
    elif dataset_name.lower() == "kvasir":
        if data_dir is None:
            data_dir = os.path.expanduser("~/afs/kvasir-dataset")
        full_ds = datasets.ImageFolder(data_dir)
        all_paths = [s[0] for s in full_ds.samples]
        all_labels = full_ds.targets
    else:
        raise ValueError(f"Unsupported ImageFolder dataset: {dataset_name}")

    train_idx, temp_idx = train_test_split(
        range(len(all_labels)), train_size=0.7, test_size=0.3,
        stratify=all_labels, random_state=seed,
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx_temp, test_idx_temp = train_test_split(
        range(len(temp_idx)), train_size=0.5, test_size=0.5,
        stratify=temp_labels, random_state=seed,
    )
    val_idx = [temp_idx[i] for i in val_idx_temp]
    test_idx = [temp_idx[i] for i in test_idx_temp]

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = np.array([all_labels[i] for i in train_idx])
    test_paths = [all_paths[i] for i in test_idx]
    test_labels = np.array([all_labels[i] for i in test_idx])

    return train_paths, train_labels, test_paths, test_labels


# ============================================================================
# DINO Collate
# ============================================================================

def dino_collate_fn(batch):
    """
    Custom collate for DINO multi-crop.
    Each sample is a list of crop tensors (2 global + N local).
    Returns a list of batched tensors, one per crop index.
    """
    n_crops = len(batch[0])
    collated = []
    for c in range(n_crops):
        collated.append(torch.stack([sample[c] for sample in batch]))
    return collated


# ============================================================================
# Linear Probe Evaluation
# ============================================================================

def linear_probe(encoder, train_imgs, train_labels, test_imgs, test_labels,
                 img_size, num_classes, device, epochs=100, batch_size=512):
    """
    Quick linear probe on frozen encoder features.
    Returns accuracy (%).
    """
    encoder.eval()
    eval_tf = EvalTransform(img_size)

    def extract_features(imgs):
        feats = []
        n_batches = (len(imgs) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                batch_np = imgs[i * batch_size:(i + 1) * batch_size]
                tensors = []
                for img_np in batch_np:
                    pil_img = Image.fromarray(img_np)
                    tensors.append(eval_tf(pil_img))
                batch_t = torch.stack(tensors).to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = encoder(batch_t)
                feats.append(out.float().cpu())
        return torch.cat(feats, dim=0)

    X_train = extract_features(train_imgs)
    X_test = extract_features(test_imgs)
    y_train = torch.from_numpy(train_labels).long()
    y_test = torch.from_numpy(test_labels).long()

    classifier = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

    classifier.train()
    for _ in range(epochs):
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        y_test_dev = y_test.to(device)
        preds = classifier(X_test_dev).argmax(dim=1)
        acc = (preds == y_test_dev).float().mean().item() * 100.0

    return acc


# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_encoder_checkpoint(wrapper, method, dataset, img_size, output_dir, tag):
    """Save backbone-only state_dict.

    File name: {method}_{dataset}_{img_size}_encoder_{tag}.pt
    Strips any wrapper prefix so keys are plain ResNet keys.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{method}_{dataset}_{img_size}_encoder_{tag}.pt"
    path = os.path.join(output_dir, fname)
    state_dict = wrapper.get_encoder_state_dict()
    torch.save(state_dict, path)
    return path


# ============================================================================
# LR Schedule Helpers
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                    steps_per_epoch, per_step=True):
    """
    Cosine decay with linear warmup.
    If per_step is True the schedule advances every step; otherwise every epoch.
    """
    if per_step:
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
    else:
        warmup_steps = warmup_epochs
        total_steps = total_epochs

    def lr_lambda(current):
        if current < warmup_steps:
            return float(current) / float(max(1, warmup_steps))
        progress = (current - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Unified Visual Encoder Pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--method", type=str, required=True,
                        choices=METHODS,
                        help="SSL method to use.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=DATASETS,
                        help="Dataset to pretrain on.")

    # Optional overrides (None means "use method default")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--momentum_coeff", type=float, default=None)

    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Use automatic mixed precision.")
    parser.add_argument("--eval_every", type=int, default=25,
                        help="Run linear probe every N epochs (0 to disable).")
    parser.add_argument("--output_dir", type=str, default="checkpoints_stage1",
                        help="Directory to save checkpoints.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root (needed for isic2019 / kvasir).")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps.")

    args = parser.parse_args()
    return args


def resolve_args(args):
    """Fill in per-method defaults for any arg the user did not override."""
    defaults = METHOD_DEFAULTS[args.method]

    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.temperature is None:
        args.temperature = defaults.get("temperature")
    if args.momentum_coeff is None:
        args.momentum_coeff = defaults.get("momentum_coeff")

    # Derived: per-method hidden/out dims (not user-settable but stored)
    args.proj_hidden = defaults["proj_hidden"]
    args.proj_out = defaults["proj_out"]

    return args


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    args = resolve_args(args)

    # ---- Seed ---------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}")
    print(f"Stage 1 Pretraining: {args.method.upper()} on {args.dataset.upper()}")
    print(f"{'=' * 80}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device       : {device}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Image size   : {args.img_size}")
    print(f"Accum steps  : {args.accum_steps}")
    print(f"AMP          : {args.use_amp}")

    # ---- LR scaling ---------------------------------------------------------
    scaled_lr = args.lr * args.batch_size / 256.0
    if args.method == "barlow" and args.dataset in SMALL_DATASETS:
        scaled_lr *= 0.3
        print(f"LR (Barlow adaptive) : {scaled_lr:.6f}")
    else:
        print(f"LR (scaled)  : {scaled_lr:.6f}")
    print(f"Weight decay : {args.weight_decay}")
    if args.temperature is not None:
        print(f"Temperature  : {args.temperature}")
    if args.momentum_coeff is not None:
        print(f"Momentum     : {args.momentum_coeff}")
    print(f"{'=' * 80}\n")

    # ---- Augmentation -------------------------------------------------------
    if args.method == "simclr":
        aug = SimCLRAugmentation(args.img_size)
    elif args.method in ("mocov3", "byol"):
        aug = MoCoAugmentation(args.img_size)
    elif args.method == "barlow":
        aug = BarlowAugmentation(args.img_size)
    elif args.method == "dino":
        aug = DINOAugmentation(args.img_size)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # ---- Dataset ------------------------------------------------------------
    is_imagefolder = args.dataset.lower() in ("isic2019", "kvasir")

    if is_imagefolder:
        train_paths, train_labels, test_paths, test_labels = \
            load_imagefolder_paths_labels(args.dataset, args.data_dir, seed=args.seed)
        train_dataset = SSLImageFolderDataset(train_paths, aug, img_size=args.img_size)
        print(f"Train samples: {len(train_dataset)}  (lazy-loaded ImageFolder)")

        # For linear probe we need numpy arrays; load lazily during eval.
        eval_train_imgs = None
        eval_test_imgs = None
    else:
        train_imgs, train_labels = load_medmnist_images(args.dataset, img_size=args.img_size)
        test_imgs, test_labels = load_medmnist_test_images(args.dataset, img_size=args.img_size)
        train_dataset = SSLImageDataset(train_imgs, aug)
        print(f"Train samples: {len(train_dataset)}  (in-memory MedMNIST)")

        eval_train_imgs = train_imgs
        eval_test_imgs = test_imgs

    # DataLoader
    collate = dino_collate_fn if args.method == "dino" else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    steps_per_epoch = len(train_loader)
    print(f"Steps/epoch  : {steps_per_epoch}")

    # ---- Model --------------------------------------------------------------
    if args.method == "simclr":
        model = SimCLRWrapper(
            proj_hidden=args.proj_hidden,
            proj_out=args.proj_out,
            temperature=args.temperature,
        )
    elif args.method == "mocov3":
        model = MoCoV3Wrapper(
            proj_hidden=args.proj_hidden,
            proj_out=args.proj_out,
            temperature=args.temperature,
            momentum_coeff=args.momentum_coeff,
        )
    elif args.method == "byol":
        model = BYOLWrapper(
            proj_hidden=args.proj_hidden,
            proj_out=args.proj_out,
            momentum_coeff=args.momentum_coeff,
        )
    elif args.method == "barlow":
        model = BarlowTwinsWrapper(
            proj_hidden=args.proj_hidden,
            proj_out=args.proj_out,
        )
    elif args.method == "dino":
        model = DINOWrapper(
            proj_hidden=args.proj_hidden,
            proj_out=args.proj_out,
            student_temp=0.1,
            teacher_temp=0.04,
            momentum_coeff=args.momentum_coeff,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params : {n_params:.2f}M\n")

    # ---- Optimizer ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay,
    )

    # ---- Scheduler ----------------------------------------------------------
    if args.method == "barlow":
        # Barlow: no warmup, CosineAnnealingLR over epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )
        scheduler_per_step = False
    elif args.method == "simclr":
        # SimCLR: epoch-wise cosine with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_epochs=10, total_epochs=args.epochs,
            steps_per_epoch=1, per_step=False,
        )
        scheduler_per_step = False
    else:
        # MoCo-v3, BYOL, DINO: step-wise cosine with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_epochs=10, total_epochs=args.epochs,
            steps_per_epoch=steps_per_epoch, per_step=True,
        )
        scheduler_per_step = True

    # ---- AMP scaler ---------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    # ---- Training -----------------------------------------------------------
    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        optimizer.zero_grad()

        for step_in_epoch, batch in enumerate(pbar, 1):
            # -- Prepare inputs -----------------------------------------------
            if args.method == "dino":
                # batch is a list of batched crop tensors
                crops = [c.to(device, non_blocking=True) for c in batch]
            else:
                # batch is a tuple (view1, view2)
                x1, x2 = batch
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

            # -- Forward + loss -----------------------------------------------
            with torch.amp.autocast("cuda",
                                    enabled=args.use_amp and device.type == "cuda"):
                if args.method == "dino":
                    loss = model(crops)
                else:
                    loss = model(x1, x2)

                loss = loss / args.accum_steps

            # -- Backward -----------------------------------------------------
            scaler.scale(loss).backward()

            if step_in_epoch % args.accum_steps == 0 or step_in_epoch == steps_per_epoch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler_per_step:
                    scheduler.step()

            running_loss += loss.item() * args.accum_steps
            n_steps += 1
            pbar.set_postfix(loss=f"{running_loss / n_steps:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Epoch-level scheduler step
        if not scheduler_per_step:
            scheduler.step()

        avg_loss = running_loss / max(n_steps, 1)
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # -- Best loss checkpoint ---------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss

        # -- Periodic linear probe eval ---------------------------------------
        do_eval = (args.eval_every > 0 and epoch % args.eval_every == 0) or \
                  (epoch == args.epochs)

        if do_eval:
            # Get encoder for eval
            if args.method == "simclr":
                eval_encoder = model.encoder
            elif args.method == "mocov3":
                eval_encoder = model.encoder_q
            elif args.method == "byol":
                eval_encoder = model.online_encoder
            elif args.method == "barlow":
                eval_encoder = model.encoder
            elif args.method == "dino":
                eval_encoder = model.student_encoder

            # Lazy-load images for ImageFolder datasets
            if is_imagefolder and eval_train_imgs is None:
                from multiprocessing import Pool, cpu_count
                print("  Loading images for linear probe evaluation...")

                def _load_img(path_and_size):
                    p, sz = path_and_size
                    return np.array(
                        Image.open(p).convert("RGB").resize((sz, sz))
                    )

                n_workers = min(cpu_count() - 1, 8)
                with Pool(n_workers) as pool:
                    eval_train_imgs = np.array(
                        list(pool.imap(
                            _load_img,
                            [(p, args.img_size) for p in train_paths],
                            chunksize=100,
                        )),
                        dtype=np.uint8,
                    )
                    eval_test_imgs = np.array(
                        list(pool.imap(
                            _load_img,
                            [(p, args.img_size) for p in test_paths],
                            chunksize=100,
                        )),
                        dtype=np.uint8,
                    )

            num_cls = NUM_CLASSES.get(args.dataset.lower(), 10)
            acc = linear_probe(
                eval_encoder,
                eval_train_imgs, train_labels,
                eval_test_imgs, test_labels,
                img_size=args.img_size,
                num_classes=num_cls,
                device=device,
            )
            print(f"  >> Linear probe accuracy: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                ckpt_path = save_encoder_checkpoint(
                    model, args.method, args.dataset, args.img_size,
                    args.output_dir, "best",
                )
                print(f"  >> New best! Saved to {ckpt_path}")

        # -- Periodic epoch checkpoint ----------------------------------------
        if epoch % 50 == 0 and epoch != args.epochs:
            save_encoder_checkpoint(
                model, args.method, args.dataset, args.img_size,
                args.output_dir, f"epoch{epoch}",
            )

    # ---- Final checkpoint ---------------------------------------------------
    final_path = save_encoder_checkpoint(
        model, args.method, args.dataset, args.img_size,
        args.output_dir, "final",
    )
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Best linear probe accuracy: {best_acc:.2f}%")

    # ---- Save summary -------------------------------------------------------
    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_base": args.lr,
        "lr_scaled": scaled_lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "momentum_coeff": args.momentum_coeff,
        "img_size": args.img_size,
        "best_linear_probe_acc": best_acc,
        "best_loss": best_loss,
        "seed": args.seed,
        "use_amp": args.use_amp,
        "accum_steps": args.accum_steps,
    }
    summary_path = os.path.join(
        args.output_dir,
        f"{args.method}_{args.dataset}_{args.img_size}_summary.json",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
