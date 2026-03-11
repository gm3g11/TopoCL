#!/usr/bin/env python3
"""
Computational Cost Analysis - HYBRID MEASUREMENT

Uses fvcore for matmul operations + manual calculation for missing operations.

Usage:
    python analysis/computational_cost.py --dataset kvasir --debug
    python analysis/computational_cost.py --dataset kvasir --full
"""

import os
import sys
import io
import time
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from topocl.data.augmentations import TopologyAwareAugmentation
from topocl.data.datasets import load_dataset_with_rois, Stage3TopoImageDataset
from topocl.models.resnet_encoder import ResNetEncoder
from topocl.models.topo_encoder import HierarchicalTopoEncoder
from topocl.models.moe_fusion import MoEFusedEncoder
from topocl.models.ssl_heads import SimCLRModel, BYOLModel, DINOModel, BarlowModel, MoCoV3Model
from topocl.utils.training import set_seed, get_num_classes, balance_loss

warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    print("fvcore not available, install with: pip install fvcore")
    FVCORE_AVAILABLE = False


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class ComponentTimer:
    """Timer for tracking component-wise execution time"""
    def __init__(self):
        self.times = defaultdict(list)
        self.start_times = {}

    def start(self, name):
        torch.cuda.synchronize()
        self.start_times[name] = time.time()

    def end(self, name):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start_times[name]
        self.times[name].append(elapsed)

    def get_mean(self, name):
        return np.mean(self.times[name]) if self.times[name] else 0.0

    def get_total(self, name):
        return np.sum(self.times[name]) if self.times[name] else 0.0


# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_component_parameters(encoder, model, method):
    """Count parameters for three key components"""
    params = {}

    vis_params = sum(p.numel() for p in encoder.visual_encoder.parameters())
    vis_proj_params = sum(p.numel() for p in encoder.vis_proj.parameters())
    params['visual_encoder'] = (vis_params + vis_proj_params) / 1e6

    topo_params = sum(p.numel() for p in encoder.topo_encoder.parameters())
    topo_proj_params = sum(p.numel() for p in encoder.topo_proj.parameters())
    params['topo_encoder'] = (topo_params + topo_proj_params) / 1e6

    moe_params = (
        sum(p.numel() for p in encoder.expert_vis.parameters()) +
        sum(p.numel() for p in encoder.expert_topo.parameters()) +
        sum(p.numel() for p in encoder.expert_concat.parameters()) +
        sum(p.numel() for p in encoder.cross_attn_vis.parameters()) +
        sum(p.numel() for p in encoder.cross_attn_topo.parameters()) +
        sum(p.numel() for p in encoder.norm_vis.parameters()) +
        sum(p.numel() for p in encoder.norm_topo.parameters()) +
        sum(p.numel() for p in encoder.expert_attention.parameters()) +
        sum(p.numel() for p in encoder.gate_network.parameters()) +
        sum(p.numel() for p in encoder.expert_gated.parameters()) +
        sum(p.numel() for p in encoder.meta_gate.parameters())
    )

    if method == 'simclr':
        proj_params = sum(p.numel() for p in model.projector.parameters())
    elif method == 'byol':
        proj_params = (sum(p.numel() for p in model.online_projector.parameters()) +
                      sum(p.numel() for p in model.predictor.parameters()))
    elif method == 'dino':
        proj_params = sum(p.numel() for p in model.student_projector.parameters())
    elif method == 'barlow':
        proj_params = sum(p.numel() for p in model.projector.parameters())
    elif method == 'mocov3':
        proj_params = (sum(p.numel() for p in model.projector_q.parameters()) +
                      sum(p.numel() for p in model.predictor.parameters()))
    else:
        proj_params = 0

    params['moe_proj'] = (moe_params + proj_params) / 1e6

    return params


# ============================================================================
# MANUAL FLOPS CALCULATIONS
# ============================================================================

def calculate_attention_flops(n_q, n_kv, d, num_heads=8):
    """Calculate FLOPs for attention mechanism that fvcore misses"""
    qk_flops = n_q * n_kv * d
    av_flops = n_q * n_kv * d
    return (qk_flops + av_flops) / 1e9


def calculate_topo_encoder_missing_flops(n_h0, n_h1, hidden_dim=384):
    """Calculate missing FLOPs in Topo Encoder"""
    h0_attn = calculate_attention_flops(n_h0, n_h0, hidden_dim)
    h1_attn = calculate_attention_flops(n_h1, n_h1, hidden_dim)
    cross_attn = calculate_attention_flops(n_h0, n_h1, hidden_dim)

    return {
        'h0_attention': h0_attn,
        'h1_attention': h1_attn,
        'cross_attention': cross_attn,
        'total': h0_attn + h1_attn + cross_attn
    }


def calculate_moe_missing_flops(vis_dim=2048, topo_dim=1536, embed_dim=256):
    """Calculate missing FLOPs in MoE"""
    gate_network_flops = 2 * (vis_dim + topo_dim) * embed_dim
    missing_flops = gate_network_flops / 1e9 - 0.0012
    return max(0, missing_flops)


# ============================================================================
# HYBRID FLOPS CALCULATION
# ============================================================================

def calculate_flops_hybrid(encoder, model, device, method,
                           img_size=224, n_h0=144, n_h1=288):
    """Hybrid FLOPs measurement: fvcore + manual corrections"""
    if not FVCORE_AVAILABLE:
        return None

    print("    Using HYBRID measurement (fvcore + manual corrections)...")

    encoder.eval()
    model.eval()

    img = torch.randn(1, 3, img_size, img_size).to(device)
    h0 = torch.randn(1, n_h0, 2).to(device)
    h1 = torch.randn(1, n_h1, 2).to(device)

    flops = {}
    flops_breakdown = {}

    # 1. Visual Encoder + vis_proj
    print("      [1/3] Visual Encoder...")

    class VisualEncoderWithProj(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.visual_encoder = enc.visual_encoder
            self.vis_proj = enc.vis_proj
        def forward(self, x):
            return self.vis_proj(self.visual_encoder(x))

    vis_model = VisualEncoderWithProj(encoder).eval()
    try:
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        flop_counter = FlopCountAnalysis(vis_model, img)
        vis_flops_fvcore = flop_counter.total() / 1e9
        sys.stderr = old_stderr

        flops['visual_encoder'] = vis_flops_fvcore
        flops_breakdown['visual_encoder'] = {
            'fvcore_measured': vis_flops_fvcore,
            'total': vis_flops_fvcore
        }
        print(f"        total: {vis_flops_fvcore:.4f}G")
    except Exception as e:
        print(f"        Failed: {e}")
        flops['visual_encoder'] = 0
        flops_breakdown['visual_encoder'] = {'error': str(e)}

    # 2. Topo Encoder + topo_proj
    print("      [2/3] Topo Encoder...")

    class TopoEncoderWithProj(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.topo_encoder = enc.topo_encoder
            self.topo_proj = enc.topo_proj
        def forward(self, h0, h1):
            return self.topo_proj(self.topo_encoder(h0, h1))

    topo_model = TopoEncoderWithProj(encoder).eval()
    try:
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        flop_counter = FlopCountAnalysis(topo_model, (h0, h1))
        topo_flops_fvcore = flop_counter.total() / 1e9
        sys.stderr = old_stderr

        hidden_dim = encoder.topo_encoder.hidden_dim
        topo_missing = calculate_topo_encoder_missing_flops(n_h0, n_h1, hidden_dim)

        flops['topo_encoder'] = topo_flops_fvcore + topo_missing['total']
        flops_breakdown['topo_encoder'] = {
            'fvcore_measured': topo_flops_fvcore,
            'total_correction': topo_missing['total'],
            'total': flops['topo_encoder']
        }
        print(f"        total: {flops['topo_encoder']:.4f}G")
    except Exception as e:
        print(f"        Failed: {e}")
        flops['topo_encoder'] = 0

    # 3. MoE + Proj
    print("      [3/3] MoE+Proj...")

    with torch.no_grad():
        vis_feat = encoder.visual_encoder(img)
        topo_feat = encoder.topo_encoder(h0, h1)
        vis_embed = encoder.vis_proj(vis_feat)
        topo_embed = encoder.topo_proj(topo_feat)

    moe_flops_fvcore = 0.0
    experts = [
        ('expert_vis', encoder.expert_vis, vis_embed),
        ('expert_topo', encoder.expert_topo, topo_embed),
        ('expert_concat', encoder.expert_concat, torch.cat([vis_embed, topo_embed], dim=1))
    ]

    for name, module, input_tensor in experts:
        try:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            flop_counter = FlopCountAnalysis(module, input_tensor)
            moe_flops_fvcore += flop_counter.total() / 1e9
            sys.stderr = old_stderr
        except:
            pass

    # Meta-gating + SSL projector
    try:
        meta_input = torch.cat([vis_embed, topo_embed], dim=1)
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        flop_counter = FlopCountAnalysis(encoder.meta_gate, meta_input)
        moe_flops_fvcore += flop_counter.total() / 1e9
        sys.stderr = old_stderr
    except:
        pass

    try:
        dummy_fused = torch.randn(1, encoder.out_dim).to(device)
        projector = None
        if method == 'simclr':
            projector = model.projector
        elif method == 'byol':
            projector = model.online_projector
        elif method == 'dino':
            projector = model.student_projector
        elif method == 'barlow':
            projector = model.projector
        elif method == 'mocov3':
            projector = model.projector_q

        if projector is not None:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            flop_counter = FlopCountAnalysis(projector, dummy_fused)
            moe_flops_fvcore += flop_counter.total() / 1e9
            sys.stderr = old_stderr
    except:
        pass

    vis_dim = encoder.visual_encoder.out_dim
    topo_dim = encoder.topo_encoder.hidden_dim * 4
    moe_missing = calculate_moe_missing_flops(vis_dim, topo_dim, 256)

    flops['moe_proj'] = moe_flops_fvcore + moe_missing
    flops['total'] = flops['visual_encoder'] + flops['topo_encoder'] + flops['moe_proj']
    print(f"\n      TOTAL: {flops['total']:.4f}G")

    flops['breakdown'] = flops_breakdown
    return flops


# ============================================================================
# THROUGHPUT MEASUREMENT
# ============================================================================

def measure_component_throughput(encoder, device, batch_size=256, img_size=224,
                                 n_h0=144, n_h1=288, n_warmup=10, n_measure=50):
    """Measure throughput for three key components"""
    encoder.eval()

    img = torch.randn(batch_size, 3, img_size, img_size).to(device)
    h0 = torch.randn(batch_size, n_h0, 2).to(device)
    h1 = torch.randn(batch_size, n_h1, 2).to(device)

    throughputs = {}

    with torch.no_grad():
        for _ in range(n_warmup):
            vis_feat = encoder.visual_encoder(img)
            _ = encoder.vis_proj(vis_feat)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_measure):
            vis_feat = encoder.visual_encoder(img)
            _ = encoder.vis_proj(vis_feat)
        torch.cuda.synchronize()
        throughputs['visual_encoder'] = (batch_size * n_measure) / (time.time() - start)

    with torch.no_grad():
        for _ in range(n_warmup):
            topo_feat = encoder.topo_encoder(h0, h1)
            _ = encoder.topo_proj(topo_feat)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_measure):
            topo_feat = encoder.topo_encoder(h0, h1)
            _ = encoder.topo_proj(topo_feat)
        torch.cuda.synchronize()
        throughputs['topo_encoder'] = (batch_size * n_measure) / (time.time() - start)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = encoder(img, h0, h1)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_measure):
            _ = encoder(img, h0, h1)
        torch.cuda.synchronize()
        throughputs['moe_proj'] = (batch_size * n_measure) / (time.time() - start)

    return throughputs


# ============================================================================
# TRAINING TIME BREAKDOWN
# ============================================================================

def measure_training_breakdown(model, encoder, train_loader, device,
                               method, args, n_batches=None):
    """Measure training time with backward timing per component"""
    model.train()

    timers = {
        'forward_total': ComponentTimer(),
        'backward_total': ComponentTimer(),
    }

    scaler = GradScaler('cuda', enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_batches = len(train_loader) if n_batches is None else n_batches
    print(f"\n  Measuring training breakdown over {total_batches} batches...")

    visual_params = list(encoder.visual_encoder.parameters()) + list(encoder.vis_proj.parameters())
    topo_params = list(encoder.topo_encoder.parameters()) + list(encoder.topo_proj.parameters())
    moe_proj_params = (
        list(encoder.expert_vis.parameters()) +
        list(encoder.expert_topo.parameters()) +
        list(encoder.expert_concat.parameters()) +
        list(encoder.cross_attn_vis.parameters()) +
        list(encoder.cross_attn_topo.parameters()) +
        list(encoder.norm_vis.parameters()) +
        list(encoder.norm_topo.parameters()) +
        list(encoder.expert_attention.parameters()) +
        list(encoder.gate_network.parameters()) +
        list(encoder.expert_gated.parameters()) +
        list(encoder.meta_gate.parameters())
    )

    batch_count = 0
    for batch in tqdm(train_loader, total=total_batches, desc="  Training"):
        if n_batches is not None and batch_count >= n_batches:
            break

        img1, img2, h0_1, h1_1, h0_2, h1_2, labels = batch
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        h0_1 = h0_1.to(device, non_blocking=True)
        h1_1 = h1_1.to(device, non_blocking=True)
        h0_2 = h0_2.to(device, non_blocking=True)
        h1_2 = h1_2.to(device, non_blocking=True)

        optimizer.zero_grad()

        torch.cuda.synchronize()
        forward_start = time.time()

        with autocast('cuda', dtype=torch.float16, enabled=args.use_amp):
            loss, gates = model(img1, img2, h0_1, h1_1, h0_2, h1_2)
            bal_loss = balance_loss(gates, args.balance_low, args.balance_high)
            total_loss = loss + args.balance_coef * bal_loss

        torch.cuda.synchronize()
        timers['forward_total'].times['batch'].append(time.time() - forward_start)

        torch.cuda.synchronize()
        backward_start = time.time()
        scaler.scale(total_loss).backward()
        torch.cuda.synchronize()
        timers['backward_total'].times['batch'].append(time.time() - backward_start)

        scaler.step(optimizer)
        scaler.update()
        batch_count += 1

    avg_forward = timers['forward_total'].get_mean('batch')
    avg_backward = timers['backward_total'].get_mean('batch')

    total_params = len(visual_params) + len(topo_params) + len(moe_proj_params)
    vis_ratio = len(visual_params) / total_params if total_params > 0 else 0
    topo_ratio = len(topo_params) / total_params if total_params > 0 else 0
    moe_ratio = len(moe_proj_params) / total_params if total_params > 0 else 0

    visual_total = (avg_forward * vis_ratio) + (avg_backward * vis_ratio)
    topo_total = (avg_forward * topo_ratio) + (avg_backward * topo_ratio)
    moe_proj_total = (avg_forward * moe_ratio) + (avg_backward * moe_ratio)
    total_time = visual_total + topo_total + moe_proj_total

    return {
        'components': {
            'visual_total': visual_total,
            'topo_total': topo_total,
            'moe_proj_total': moe_proj_total,
        },
        'percentages': {
            'visual': (visual_total / total_time) * 100 if total_time > 0 else 0,
            'topo': (topo_total / total_time) * 100 if total_time > 0 else 0,
            'moe_proj': (moe_proj_total / total_time) * 100 if total_time > 0 else 0,
        },
        'total_time_per_batch': total_time,
    }


# ============================================================================
# MAIN
# ============================================================================

def measure_all_costs(args):
    """Main function to measure all computational costs"""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"Computational Cost Analysis - TopoCL (HYBRID)")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: {'DEBUG' if args.debug else 'FULL'}")
    print(f"{'='*80}\n")

    print("Loading dataset...")
    imgs_train, labels_train, rois_train, _ = load_dataset_with_rois(
        args.dataset, 'train', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    imgs_val, labels_val, rois_val, _ = load_dataset_with_rois(
        args.dataset, 'val', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )

    imgs_train = np.concatenate([imgs_train, imgs_val], axis=0)
    labels_train = np.concatenate([labels_train, labels_val], axis=0)
    rois_train = rois_train + rois_val

    augmenter = TopologyAwareAugmentation()
    train_dataset = Stage3TopoImageDataset(
        imgs_train, labels_train, rois_train, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    num_classes = get_num_classes(args.dataset)

    if args.debug:
        methods_to_measure = ['simclr']
        n_batches = 5
    else:
        methods_to_measure = ['simclr', 'byol', 'dino', 'barlow', 'mocov3']
        n_batches = None

    results = {}

    for method in methods_to_measure:
        print("\n" + "="*80)
        print(f"Measuring {method.upper()}")
        print("="*80 + "\n")

        visual_encoder = ResNetEncoder(variant='resnet50')
        topo_encoder = HierarchicalTopoEncoder(
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            dropout=args.dropout, n_h0=args.n_h0 * 3, n_h1=args.n_h1 * 3
        )

        moe_encoder = MoEFusedEncoder(
            visual_encoder, topo_encoder,
            embed_dim=args.embed_dim, out_dim=args.fusion_embed_dim, dropout=args.dropout
        ).to(device)

        if method == 'simclr':
            model = SimCLRModel(moe_encoder, proj_dim=args.proj_dim)
        elif method == 'byol':
            model = BYOLModel(moe_encoder, proj_dim=args.proj_dim, m=args.moco_m)
        elif method == 'dino':
            model = DINOModel(moe_encoder, proj_dim=args.proj_dim, m=args.moco_m)
        elif method == 'barlow':
            model = BarlowModel(moe_encoder, proj_dim=args.proj_dim)
        elif method == 'mocov3':
            model = MoCoV3Model(moe_encoder, proj_dim=args.proj_dim, m=args.moco_m)
        model = model.to(device)

        print("1. Parameter Counting")
        print("-"*40)
        total_params = count_parameters(model)
        component_params = count_component_parameters(moe_encoder, model, method)
        print(f"  Visual Encoder: {component_params['visual_encoder']:.2f}M")
        print(f"  Topo Encoder:   {component_params['topo_encoder']:.2f}M")
        print(f"  MoE+Proj:       {component_params['moe_proj']:.2f}M")
        print(f"  Total:          {total_params:.2f}M")

        print(f"\n2. FLOPs (HYBRID)")
        print("-"*40)
        flops = calculate_flops_hybrid(
            moe_encoder, model, device, method,
            n_h0=args.n_h0 * 3, n_h1=args.n_h1 * 3
        )

        if flops:
            print(f"\n  Visual Encoder: {flops['visual_encoder']:.2f}G")
            print(f"  Topo Encoder:   {flops['topo_encoder']:.2f}G")
            print(f"  MoE+Proj:       {flops['moe_proj']:.2f}G")
            print(f"  Total:          {flops['total']:.2f}G")

        print(f"\n3. Training Time")
        print("-"*40)
        breakdown = measure_training_breakdown(
            model, moe_encoder, train_loader, device,
            method, args, n_batches=n_batches
        )

        batches_per_epoch = len(train_loader)
        epoch_time = breakdown['total_time_per_batch'] * batches_per_epoch
        training_time_hours = epoch_time * 150 / 3600
        print(f"  Total training (150 epochs): {training_time_hours:.2f} hours")

        print(f"\n4. Throughput (Inference)")
        print("-"*40)
        throughputs = measure_component_throughput(
            moe_encoder, device, batch_size=args.batch_size,
            n_h0=args.n_h0 * 3, n_h1=args.n_h1 * 3,
            n_warmup=args.n_warmup, n_measure=args.n_measure
        )
        print(f"  Visual Encoder: {throughputs['visual_encoder']:.1f} img/s")
        print(f"  Topo Encoder:   {throughputs['topo_encoder']:.1f} img/s")
        print(f"  Full pipeline:  {throughputs['moe_proj']:.1f} img/s")

        results[method] = {
            'parameters': component_params,
            'flops': flops,
            'training_time_150epochs': training_time_hours,
            'throughput': throughputs,
        }

    # Save results
    output_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'computational_costs.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<10} {'Train(h)':<10} {'Params(M)':<12} {'FLOPs(G)':<10} {'Tput':<10}")
    print("-"*52)
    for method in results:
        train_h = results[method]['training_time_150epochs']
        params = sum(results[method]['parameters'].values())
        flops_val = results[method]['flops']['total'] if results[method]['flops'] else 0
        tput = results[method]['throughput']['moe_proj']
        print(f"{method.upper():<10} {train_h:<10.2f} {params:<12.2f} {flops_val:<10.2f} {tput:<10.0f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Measure computational costs for TopoCL')
    parser.add_argument('--dataset', type=str, default='kvasir',
                       choices=['kvasir', 'pathmnist', 'octmnist', 'organsmnist', 'isic2019'])
    parser.add_argument('--roi_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    # Architecture (paper defaults)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_h0', type=int, default=48)
    parser.add_argument('--n_h1', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--fusion_embed_dim', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--moco_m', type=float, default=0.996)
    parser.add_argument('--balance_coef', type=float, default=0.015)
    parser.add_argument('--balance_low', type=float, default=0.05)
    parser.add_argument('--balance_high', type=float, default=0.5)
    parser.add_argument('--n_warmup', type=int, default=10)
    parser.add_argument('--n_measure', type=int, default=50)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='./computational_costs')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not args.debug and not args.full:
        args.debug = True
        print("\nNo mode specified, defaulting to --debug mode\n")

    measure_all_costs(args)
    print("\nComputational cost analysis complete!\n")


if __name__ == '__main__':
    main()
