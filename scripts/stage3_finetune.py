#!/usr/bin/env python3
"""
Stage 3: Joint MoE Fine-tuning

Trains 5 independent SSL models (SimCLR, BYOL, DINO, Barlow Twins, MoCo-v3)
that each wrap a MoEFusedEncoder (visual + topology). Loads pretrained Stage 1
(visual backbone) and Stage 2 (topology encoder) weights. Uses differential
learning rates, expert balance loss, periodic linear probe evaluation, and
early stopping.
"""

import os
import sys
import time
import argparse
import yaml
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from topocl.data.augmentations import TopologyAwareAugmentation
from topocl.data.datasets import load_dataset_with_rois, Stage3TopoImageDataset
from topocl.models.resnet_encoder import ResNetEncoder, detect_resnet_variant
from topocl.models.topo_encoder import HierarchicalTopoEncoder
from topocl.models.moe_fusion import MoEFusedEncoder
from topocl.models.ssl_heads import SimCLRModel, BYOLModel, DINOModel, BarlowModel, MoCoV3Model
from topocl.losses.contrastive import info_nce_loss, supcon_loss
from topocl.utils.training import (
    set_seed, get_num_classes, balance_loss,
    print_expert_statistics, get_cosine_schedule_with_warmup,
)
from topocl.utils.evaluation import linear_probe_evaluation


# ============================================================================
# Checkpoint loading helpers
# ============================================================================

def load_checkpoint_config(config_path):
    """Load checkpoint paths from a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_stage1_paths(args, method_name):
    """
    Resolve Stage 1 checkpoint path for a given SSL method.
    Tries YAML config first, then falls back to --stage1_dir.
    """
    if args.checkpoint_config and os.path.exists(args.checkpoint_config):
        config = load_checkpoint_config(args.checkpoint_config)
        stage1 = config.get('stage1', {})
        dataset_key = args.dataset.lower()
        dataset_paths = stage1.get(dataset_key, {})
        path = dataset_paths.get(method_name, None)
        if path and os.path.exists(path):
            return path

    # Fallback: look in --stage1_dir
    if args.stage1_dir:
        candidates = [
            os.path.join(args.stage1_dir, args.dataset, method_name, 'best_encoder.pt'),
            os.path.join(args.stage1_dir, args.dataset, method_name, 'checkpoint.pt'),
            os.path.join(args.stage1_dir, method_name, args.dataset, 'best_encoder.pt'),
            os.path.join(args.stage1_dir, method_name, args.dataset, 'checkpoint.pt'),
            os.path.join(args.stage1_dir, f'{method_name}_{args.dataset}.pt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    return None


def resolve_stage2_path(args):
    """
    Resolve Stage 2 checkpoint path.
    Tries YAML config first, then falls back to --stage2_dir.
    """
    if args.checkpoint_config and os.path.exists(args.checkpoint_config):
        config = load_checkpoint_config(args.checkpoint_config)
        stage2 = config.get('stage2', {})
        dataset_key = args.dataset.lower()
        path = stage2.get(dataset_key, None)
        if path and os.path.exists(path):
            return path

    if args.stage2_dir:
        candidates = [
            os.path.join(args.stage2_dir, args.dataset, 'best_encoder.pt'),
            os.path.join(args.stage2_dir, args.dataset, 'simclr_ssl', 'best_encoder.pt'),
            os.path.join(args.stage2_dir, args.dataset, 'supcon', 'best_encoder.pt'),
            os.path.join(args.stage2_dir, args.dataset, 'supervised_ce', 'best_encoder.pt'),
            os.path.join(args.stage2_dir, f'{args.dataset}_topo_encoder.pt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    return None


def should_strip_m_prefix(method_name, dataset_name):
    """
    Determine whether to strip the 'm.' prefix from state dict keys.

    BYOL and MoCo-v3 always strip because their Stage 1 checkpoints use an
    online/query encoder wrapper that prefixes keys with 'm.'.
    SimCLR strips for MedMNIST datasets (the Stage 1 code wraps the encoder).
    Barlow strips for organsmnist.
    """
    method_lower = method_name.lower()
    dataset_lower = dataset_name.lower()
    medmnist_datasets = {'pathmnist', 'octmnist', 'organamnist', 'organsmnist'}

    if method_lower in ('byol', 'mocov3'):
        return True
    if method_lower == 'simclr' and dataset_lower in medmnist_datasets:
        return True
    if method_lower == 'barlow' and dataset_lower == 'organsmnist':
        return True
    return False


def strip_m_prefix(state_dict):
    """Remove 'm.' prefix from all keys in a state dict."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('m.'):
            new_sd[k[2:]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_visual_encoder_weights(visual_encoder, ckpt_path, method_name, dataset_name):
    """
    Load Stage 1 visual encoder weights from a checkpoint.
    Handles key-prefix stripping and auto-detection of the state dict location.
    """
    print(f"    Loading Stage 1 ({method_name}) from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Try common keys where the encoder state dict might be stored
    possible_keys = [
        'encoder_state_dict',
        'model_state_dict',
        'state_dict',
        'backbone',
        'encoder',
    ]

    sd = None
    for key in possible_keys:
        if key in ckpt:
            sd = ckpt[key]
            print(f"      Found state dict under key '{key}' ({len(sd)} keys)")
            break

    if sd is None:
        # Maybe the checkpoint is the raw state dict
        if isinstance(ckpt, dict) and any('conv1' in k or 'layer1' in k for k in ckpt.keys()):
            sd = ckpt
            print(f"      Using raw checkpoint as state dict ({len(sd)} keys)")
        else:
            print(f"      WARNING: Could not find encoder state dict in checkpoint")
            return False

    # Decide whether to strip 'm.' prefix
    if should_strip_m_prefix(method_name, dataset_name):
        has_m_prefix = any(k.startswith('m.') for k in sd.keys())
        if has_m_prefix:
            sd = strip_m_prefix(sd)
            print(f"      Stripped 'm.' prefix ({method_name}/{dataset_name})")

    # Filter to keys that match the visual encoder
    encoder_keys = set(visual_encoder.state_dict().keys())
    filtered_sd = {k: v for k, v in sd.items() if k in encoder_keys}

    if len(filtered_sd) == 0:
        print(f"      WARNING: No matching keys found. Encoder keys sample: {list(encoder_keys)[:5]}")
        print(f"      Checkpoint keys sample: {list(sd.keys())[:5]}")
        return False

    missing, unexpected = visual_encoder.load_state_dict(filtered_sd, strict=False)
    n_loaded = len(filtered_sd)
    n_total = len(encoder_keys)
    print(f"      Loaded {n_loaded}/{n_total} keys, {len(missing)} missing, {len(unexpected)} unexpected")
    return True


def load_topo_encoder_weights(topo_encoder, ckpt_path):
    """Load Stage 2 topology encoder weights."""
    print(f"    Loading Stage 2 topo encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'encoder_state_dict' in ckpt:
        sd = ckpt['encoder_state_dict']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    # Filter to topo encoder keys
    topo_keys = set(topo_encoder.state_dict().keys())
    filtered_sd = {k: v for k, v in sd.items() if k in topo_keys}

    if len(filtered_sd) == 0:
        # Try stripping a prefix (e.g. 'encoder.')
        stripped = {}
        for k, v in sd.items():
            for prefix in ['encoder.', 'topo_encoder.', 'module.']:
                if k.startswith(prefix):
                    new_k = k[len(prefix):]
                    if new_k in topo_keys:
                        stripped[new_k] = v
                        break
        filtered_sd = stripped

    if len(filtered_sd) > 0:
        missing, unexpected = topo_encoder.load_state_dict(filtered_sd, strict=False)
        n_loaded = len(filtered_sd)
        n_total = len(topo_keys)
        print(f"      Loaded {n_loaded}/{n_total} keys, {len(missing)} missing")
        return True
    else:
        print(f"      WARNING: No matching keys found for topo encoder")
        return False


# ============================================================================
# Build SSL models
# ============================================================================

def build_ssl_model(method_name, visual_encoder, topo_encoder, args):
    """
    Build a complete SSL model for Stage 3:
      visual_encoder + topo_encoder -> MoEFusedEncoder -> SSL head
    """
    moe = MoEFusedEncoder(
        visual_encoder=visual_encoder,
        topo_encoder=topo_encoder,
        embed_dim=args.embed_dim,
        out_dim=args.fusion_embed_dim,
        dropout=args.dropout,
    )

    if method_name == 'simclr':
        model = SimCLRModel(moe, proj_dim=args.proj_dim)
    elif method_name == 'byol':
        model = BYOLModel(moe, proj_dim=args.proj_dim, m=args.moco_m)
    elif method_name == 'dino':
        model = DINOModel(moe, proj_dim=args.proj_dim, m=args.moco_m)
    elif method_name == 'barlow':
        model = BarlowModel(moe, proj_dim=args.proj_dim)
    elif method_name == 'mocov3':
        model = MoCoV3Model(moe, proj_dim=args.proj_dim, m=args.moco_m)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return model


def get_moe_encoder_from_model(model, method_name):
    """Extract the MoE encoder from an SSL model wrapper."""
    if method_name == 'simclr':
        return model.encoder
    elif method_name == 'byol':
        return model.online_encoder
    elif method_name == 'dino':
        return model.student_encoder
    elif method_name == 'barlow':
        return model.encoder
    elif method_name == 'mocov3':
        return model.encoder_q
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================================
# Optimizer with differential learning rates
# ============================================================================

def build_optimizer(model, method_name, args):
    """
    Build optimizer with differential learning rates:
      - lr_img for visual encoder parameters
      - lr_topo for topology encoder parameters
      - lr_fusion for everything else (projectors, gating, SSL heads)
    """
    moe = get_moe_encoder_from_model(model, method_name)
    visual_params = set(id(p) for p in moe.visual_encoder.parameters())
    topo_params = set(id(p) for p in moe.topo_encoder.parameters())

    visual_group = []
    topo_group = []
    fusion_group = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in visual_params:
            visual_group.append(param)
        elif pid in topo_params:
            topo_group.append(param)
        else:
            fusion_group.append(param)

    param_groups = [
        {'params': visual_group, 'lr': args.lr_img, 'name': 'visual'},
        {'params': topo_group, 'lr': args.lr_topo, 'name': 'topo'},
        {'params': fusion_group, 'lr': args.lr_fusion, 'name': 'fusion'},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    n_vis = sum(p.numel() for p in visual_group)
    n_topo = sum(p.numel() for p in topo_group)
    n_fusion = sum(p.numel() for p in fusion_group)
    print(f"    Param groups: visual={n_vis:,} (lr={args.lr_img}), "
          f"topo={n_topo:,} (lr={args.lr_topo}), "
          f"fusion={n_fusion:,} (lr={args.lr_fusion})")

    return optimizer


# ============================================================================
# Training step for a single method
# ============================================================================

def train_step(model, method_name, batch, optimizer, scaler, args, device):
    """
    Run one training step for a single SSL method.

    Returns:
        ssl_loss: float, the raw SSL loss
        bal_loss: float, the expert balance loss
        gates: (B, 5) tensor of gate weights (detached)
    """
    img1, img2, h0_1, h1_1, h0_2, h1_2, labels = batch
    img1 = img1.to(device)
    img2 = img2.to(device)
    h0_1 = h0_1.to(device)
    h1_1 = h1_1.to(device)
    h0_2 = h0_2.to(device)
    h1_2 = h1_2.to(device)
    labels = labels.to(device).long()

    optimizer.zero_grad()

    with autocast('cuda', dtype=torch.float16, enabled=args.use_amp):
        if method_name in ('simclr', 'mocov3'):
            ssl_loss, gates = model(
                img1, img2, h0_1, h1_1, h0_2, h1_2,
                temperature=args.temperature,
            )
        else:
            ssl_loss, gates = model(img1, img2, h0_1, h1_1, h0_2, h1_2)

        bal = balance_loss(
            gates,
            low_threshold=args.balance_low,
            high_threshold=args.balance_high,
        )
        total_loss = ssl_loss + args.balance_coef * bal

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return ssl_loss.item(), bal.item(), gates.detach()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 3: Joint MoE Fine-tuning')

    # Data
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--roi_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    # Learning rates
    parser.add_argument('--lr_img', type=float, default=1e-5)
    parser.add_argument('--lr_topo', type=float, default=2e-4)
    parser.add_argument('--lr_fusion', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_h0', type=int, default=48)
    parser.add_argument('--n_h1', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--fusion_embed_dim', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=256)

    # SSL
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--moco_m', type=float, default=0.996)

    # Expert balance
    parser.add_argument('--balance_coef', type=float, default=0.015)
    parser.add_argument('--balance_low', type=float, default=0.05)
    parser.add_argument('--balance_high', type=float, default=0.5)

    # Checkpoint loading
    parser.add_argument('--stage1_dir', type=str, default=None)
    parser.add_argument('--stage2_dir', type=str, default=None)
    parser.add_argument('--checkpoint_config', type=str, default=None,
                        help='Path to YAML file with checkpoint paths')

    # Other
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--probe_freq', type=int, default=30)
    parser.add_argument('--monitor_freq', type=int, default=20)
    parser.add_argument('--early_stop_patience', type=int, default=40)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = get_num_classes(args.dataset)
    ssl_methods = ['simclr', 'byol', 'dino', 'barlow', 'mocov3']

    print("=" * 80)
    print("Stage 3: Joint MoE Fine-tuning")
    print("=" * 80)
    print(f"  Dataset: {args.dataset} ({num_classes} classes)")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  LR: img={args.lr_img}, topo={args.lr_topo}, fusion={args.lr_fusion}")
    print(f"  Architecture: hidden={args.hidden_dim}, heads={args.num_heads}, "
          f"embed={args.embed_dim}, fusion={args.fusion_embed_dim}")
    print(f"  SSL: temp={args.temperature}, moco_m={args.moco_m}")
    print(f"  Balance: coef={args.balance_coef}, low={args.balance_low}, high={args.balance_high}")
    print(f"  AMP: {args.use_amp}, Grad clip: {args.grad_clip}")
    print(f"  Probe freq: {args.probe_freq}, Monitor freq: {args.monitor_freq}")
    print(f"  Early stop patience: {args.early_stop_patience}")
    print(f"  Debug: {args.debug}")

    # -----------------------------------------------------------------------
    # Default checkpoint config
    # -----------------------------------------------------------------------
    if args.checkpoint_config is None:
        default_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'checkpoint_paths.yaml'
        )
        if os.path.exists(default_config):
            args.checkpoint_config = default_config
            print(f"\n  Using default checkpoint config: {default_config}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    # Train + val merged for training
    print("  Loading train split...")
    train_imgs, train_labels, train_rois, train_sam = load_dataset_with_rois(
        args.dataset, 'train', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(f"    Train: {train_imgs.shape}, SAM={train_sam}")

    print("  Loading val split...")
    val_imgs, val_labels, val_rois, val_sam = load_dataset_with_rois(
        args.dataset, 'val', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(f"    Val: {val_imgs.shape}, SAM={val_sam}")

    # Merge train + val
    merged_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    merged_labels = np.concatenate([train_labels, val_labels], axis=0)
    merged_rois = train_rois + val_rois
    print(f"    Merged train+val: {merged_imgs.shape}")

    # Test for evaluation
    print("  Loading test split...")
    test_imgs, test_labels, test_rois, test_sam = load_dataset_with_rois(
        args.dataset, 'test', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(f"    Test: {test_imgs.shape}, SAM={test_sam}")

    # Create dataset and loader
    augmenter = TopologyAwareAugmentation()

    train_dataset = Stage3TopoImageDataset(
        merged_imgs, merged_labels, merged_rois, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    print(f"  Train loader: {len(train_loader)} batches of {args.batch_size}")

    # -----------------------------------------------------------------------
    # Resolve Stage 2 checkpoint
    # -----------------------------------------------------------------------
    stage2_path = resolve_stage2_path(args)
    if stage2_path:
        print(f"\n  Stage 2 checkpoint: {stage2_path}")
    else:
        print("\n  WARNING: No Stage 2 checkpoint found. Topo encoder will use random init.")

    # -----------------------------------------------------------------------
    # Detect ResNet variant from a Stage 1 checkpoint (use simclr as reference)
    # -----------------------------------------------------------------------
    resnet_variant = 'resnet50'  # default
    for probe_method in ssl_methods:
        probe_path = resolve_stage1_paths(args, probe_method)
        if probe_path:
            probe_ckpt = torch.load(probe_path, map_location='cpu')
            for key in ['encoder_state_dict', 'model_state_dict', 'state_dict']:
                if key in probe_ckpt:
                    sd = probe_ckpt[key]
                    if should_strip_m_prefix(probe_method, args.dataset):
                        if any(k.startswith('m.') for k in sd.keys()):
                            sd = strip_m_prefix(sd)
                    resnet_variant = detect_resnet_variant(sd)
                    print(f"  Auto-detected ResNet variant: {resnet_variant} (from {probe_method})")
                    break
            break

    # -----------------------------------------------------------------------
    # Build 5 SSL models
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Building 5 SSL models...")
    print("=" * 80)

    # PH dimensions: n_h0*3 and n_h1*3 due to 3 scales in persistence computation
    n_h0_total = args.n_h0 * 3
    n_h1_total = args.n_h1 * 3

    models = {}
    optimizers = {}
    schedulers = {}
    scalers = {}

    for method in ssl_methods:
        print(f"\n  [{method.upper()}]")

        # Build fresh visual and topo encoders for each method
        visual_enc = ResNetEncoder(variant=resnet_variant)
        topo_enc = HierarchicalTopoEncoder(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            n_h0=n_h0_total,
            n_h1=n_h1_total,
        )

        # Load Stage 1 visual weights
        s1_path = resolve_stage1_paths(args, method)
        if s1_path:
            load_visual_encoder_weights(visual_enc, s1_path, method, args.dataset)
        else:
            print(f"    WARNING: No Stage 1 checkpoint for {method}. Using random init.")

        # Load Stage 2 topo weights
        if stage2_path:
            load_topo_encoder_weights(topo_enc, stage2_path)

        # Build complete SSL model
        model = build_ssl_model(method, visual_enc, topo_enc, args)
        model = model.to(device)
        models[method] = model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Total params: {total_params:,}, Trainable: {trainable_params:,}")

        # Optimizer
        optimizer = build_optimizer(model, method, args)
        optimizers[method] = optimizer

        # Scheduler
        num_steps = args.epochs * len(train_loader)
        warmup_steps = min(5 * len(train_loader), num_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
        schedulers[method] = scheduler

        # Scaler
        scalers[method] = GradScaler('cuda', enabled=args.use_amp)

    # -----------------------------------------------------------------------
    # Save directory
    # -----------------------------------------------------------------------
    if args.save_dir is None:
        args.save_dir = os.path.join('checkpoints', 'stage3', args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch = 1
    best_probe_accs = {m: 0.0 for m in ssl_methods}
    epochs_without_improvement = 0

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_probe_accs = resume_ckpt.get('best_probe_accs', best_probe_accs)
        epochs_without_improvement = resume_ckpt.get('epochs_without_improvement', 0)

        for method in ssl_methods:
            if f'{method}_model' in resume_ckpt:
                models[method].load_state_dict(resume_ckpt[f'{method}_model'])
            if f'{method}_optimizer' in resume_ckpt:
                optimizers[method].load_state_dict(resume_ckpt[f'{method}_optimizer'])
            if f'{method}_scaler' in resume_ckpt:
                scalers[method].load_state_dict(resume_ckpt[f'{method}_scaler'])
        print(f"  Resumed at epoch {start_epoch}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    print("=" * 80)

    global_step = (start_epoch - 1) * len(train_loader)
    t0_total = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0_epoch = time.time()

        # Accumulate losses and gates per method
        epoch_losses = {m: [] for m in ssl_methods}
        epoch_bal_losses = {m: [] for m in ssl_methods}
        epoch_gates = {m: [] for m in ssl_methods}

        # Set all models to train
        for method in ssl_methods:
            models[method].train()

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1

            for method in ssl_methods:
                ssl_loss, bal_loss_val, gates = train_step(
                    models[method], method, batch,
                    optimizers[method], scalers[method], args, device,
                )
                schedulers[method].step()

                epoch_losses[method].append(ssl_loss)
                epoch_bal_losses[method].append(bal_loss_val)
                epoch_gates[method].append(gates.cpu())

                if args.debug and (batch_idx + 1) % 10 == 0:
                    lr_groups = [pg['lr'] for pg in optimizers[method].param_groups]
                    print(f"    [{method}] batch {batch_idx+1}/{len(train_loader)} | "
                          f"ssl={ssl_loss:.4f} | bal={bal_loss_val:.4f} | "
                          f"lrs={[f'{lr:.2e}' for lr in lr_groups]}")

        # Epoch summary
        elapsed = time.time() - t0_epoch
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")

        for method in ssl_methods:
            avg_ssl = np.mean(epoch_losses[method])
            avg_bal = np.mean(epoch_bal_losses[method])
            lr_vis = optimizers[method].param_groups[0]['lr']
            lr_topo = optimizers[method].param_groups[1]['lr']
            lr_fuse = optimizers[method].param_groups[2]['lr']
            print(f"  {method:>8s}: ssl={avg_ssl:.4f}  bal={avg_bal:.4f}  "
                  f"lr=[{lr_vis:.1e},{lr_topo:.1e},{lr_fuse:.1e}]")

        # ---------------------------------------------------------------
        # Expert monitoring
        # ---------------------------------------------------------------
        if epoch % args.monitor_freq == 0:
            for method in ssl_methods:
                all_gates = torch.cat(epoch_gates[method], dim=0)
                print_expert_statistics(epoch, all_gates, method)

        # ---------------------------------------------------------------
        # Periodic linear probe evaluation
        # ---------------------------------------------------------------
        if epoch % args.probe_freq == 0:
            print(f"\n{'=' * 60}")
            print(f"Linear Probe Evaluation (epoch {epoch})")
            print(f"{'=' * 60}")

            any_improved = False

            for method in ssl_methods:
                moe_enc = get_moe_encoder_from_model(models[method], method)
                probe_acc = linear_probe_evaluation(
                    encoder=moe_enc,
                    imgs_train=merged_imgs,
                    labels_train=merged_labels,
                    rois_train=merged_rois,
                    imgs_test=test_imgs,
                    labels_test=test_labels,
                    rois_test=test_rois,
                    n_h0=n_h0_total,
                    n_h1=n_h1_total,
                    num_classes=num_classes,
                    device=device,
                    epochs=100,
                    batch_size=512,
                )
                improved = ""
                if probe_acc > best_probe_accs[method]:
                    best_probe_accs[method] = probe_acc
                    any_improved = True
                    improved = " (NEW BEST)"

                    # Save best checkpoint for this method
                    ckpt = {
                        'epoch': epoch,
                        'method': method,
                        'dataset': args.dataset,
                        'probe_acc': probe_acc,
                        'encoder_state_dict': moe_enc.state_dict(),
                        'model_state_dict': models[method].state_dict(),
                        'args': vars(args),
                    }
                    ckpt_path = os.path.join(args.save_dir, f'best_{method}.pt')
                    torch.save(ckpt, ckpt_path)

                print(f"  {method:>8s}: {probe_acc:.2f}%  (best: {best_probe_accs[method]:.2f}%){improved}")

            if any_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += args.probe_freq

            print(f"\n  Epochs without improvement: {epochs_without_improvement}")

            # Early stopping
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\n  Early stopping triggered (patience={args.early_stop_patience})")
                break

        # ---------------------------------------------------------------
        # Periodic full checkpoint for resume
        # ---------------------------------------------------------------
        if epoch % args.probe_freq == 0 or epoch == args.epochs:
            resume_ckpt = {
                'epoch': epoch,
                'best_probe_accs': best_probe_accs,
                'epochs_without_improvement': epochs_without_improvement,
                'args': vars(args),
            }
            for method in ssl_methods:
                resume_ckpt[f'{method}_model'] = models[method].state_dict()
                resume_ckpt[f'{method}_optimizer'] = optimizers[method].state_dict()
                resume_ckpt[f'{method}_scaler'] = scalers[method].state_dict()

            resume_path = os.path.join(args.save_dir, 'resume_checkpoint.pt')
            torch.save(resume_ckpt, resume_path)
            print(f"  Resume checkpoint saved to: {resume_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_time = time.time() - t0_total
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"  Total time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
    print(f"  Save directory: {args.save_dir}")
    print(f"\n  Best linear probe accuracies:")
    for method in ssl_methods:
        print(f"    {method:>8s}: {best_probe_accs[method]:.2f}%")

    best_overall = max(best_probe_accs, key=best_probe_accs.get)
    print(f"\n  Overall best: {best_overall} ({best_probe_accs[best_overall]:.2f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
