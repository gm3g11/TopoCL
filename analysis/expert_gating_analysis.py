#!/usr/bin/env python3
"""
Generate Expert Gating Analysis Figures

Creates:
1. Individual figures for each dataset-method combination (25 total)
2. Summary figures combining methods per dataset (5 total)
3. Organized output for main paper + supplementary material

Usage:
    python analysis/expert_gating_analysis.py --results_dir stage3_results
"""

import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import json

from topocl.data.augmentations import TopologyAwareAugmentation
from topocl.data.datasets import load_dataset_with_rois, Stage3TopoImageDataset
from topocl.models.resnet_encoder import ResNetEncoder
from topocl.models.topo_encoder import HierarchicalTopoEncoder
from topocl.models.moe_fusion import MoEFusedEncoder

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")

DATASET_INFO = {
    'pathmnist': {
        'name': 'PathMNIST', 'modality': 'Colon Pathology',
        'classes': [f'Tissue-{i}' for i in range(9)]
    },
    'octmnist': {
        'name': 'OCTMNIST', 'modality': 'Retinal OCT',
        'classes': ['Normal', 'CNV', 'DME', 'Drusen']
    },
    'organsmnist': {
        'name': 'OrganSMNIST', 'modality': 'Abdominal CT',
        'classes': [f'Organ-{i}' for i in range(11)]
    },
    'isic2019': {
        'name': 'ISIC2019', 'modality': 'Dermoscopy',
        'classes': ['Melanoma', 'Melanocytic nevus', 'BCC', 'AK', 'BKL', 'DF', 'Vascular', 'SCC']
    },
    'kvasir': {
        'name': 'Kvasir', 'modality': 'GI Endoscopy',
        'classes': ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis',
                   'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
    }
}


def load_model_and_encoder(checkpoint_path, device='cuda',
                           hidden_dim=384, num_heads=4, dropout=0.15,
                           n_h0=48, n_h1=96, embed_dim=256, fusion_embed_dim=512):
    """Load trained model from checkpoint"""
    visual_encoder = ResNetEncoder(variant='resnet50')
    topo_encoder = HierarchicalTopoEncoder(
        hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
        n_h0=n_h0 * 3, n_h1=n_h1 * 3
    )

    moe_encoder = MoEFusedEncoder(
        visual_encoder, topo_encoder,
        embed_dim=embed_dim, out_dim=fusion_embed_dim, dropout=dropout
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'encoder_state_dict' in checkpoint:
            moe_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        acc = checkpoint.get('probe_acc', 0)
        return moe_encoder, epoch, acc
    else:
        print(f"  Error: Checkpoint not found: {checkpoint_path}")
        return None, 0, 0


def collect_gating_data(encoder, test_loader, device='cuda'):
    """Collect gating weights, labels, and images"""
    encoder.eval()
    all_gates, all_labels, all_images = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Collecting gating data', leave=False):
            img1, img2, h0_1, h1_1, h0_2, h1_2, labels = batch
            img1 = img1.to(device, non_blocking=True)
            h0_1 = h0_1.to(device, non_blocking=True)
            h1_1 = h1_1.to(device, non_blocking=True)

            _, gates = encoder(img1, h0_1, h1_1)

            all_gates.append(gates.cpu().numpy())
            all_labels.append(labels.numpy())
            all_images.append(img1.cpu().numpy())

    return np.concatenate(all_gates), np.concatenate(all_labels), np.concatenate(all_images)


def plot_gating_analysis_single(all_gates, all_labels, all_images,
                                dataset_name, method_name, class_names, save_path):
    """Generate expert gating analysis for ONE dataset-method combination"""
    expert_names = ['Vis-Only', 'Topo-Only', 'Concat', 'Gated', 'Cross-Attn']
    num_experts = 5
    num_classes = len(np.unique(all_labels))

    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1.2, 2], wspace=0.3)

    # (a) Bar Chart
    ax1 = fig.add_subplot(gs[0])
    mean_gates = all_gates.mean(axis=0)
    std_gates = all_gates.std(axis=0)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ax1.bar(range(num_experts), mean_gates, yerr=std_gates,
            color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Average Weight', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(num_experts))
    ax1.set_xticklabels(expert_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim([0, max(mean_gates) * 1.3])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_title('(a) Average Expert Usage', fontsize=13, fontweight='bold', pad=10)

    for i, (v, s) in enumerate(zip(mean_gates, std_gates)):
        ax1.text(i, v + s + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (b) Heatmap
    ax2 = fig.add_subplot(gs[1])
    class_gates = np.zeros((num_classes, num_experts))
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_gates[c] = all_gates[mask].mean(axis=0)

    im = ax2.imshow(class_gates, aspect='auto', cmap='YlOrRd', vmin=0, vmax=class_gates.max())
    ax2.set_xticks(range(num_experts))
    ax2.set_xticklabels(expert_names, rotation=45, ha='right', fontsize=10)
    ax2.set_yticks(range(num_classes))
    display_names = [n[:20] + '...' if len(n) > 20 else n for n in class_names[:num_classes]]
    ax2.set_yticklabels(display_names, fontsize=9)
    ax2.set_title('(b) Expert Weights per Class', fontsize=13, fontweight='bold', pad=10)

    for i in range(num_classes):
        for j in range(num_experts):
            value = class_gates[i, j]
            color = 'white' if value > class_gates.max() * 0.5 else 'black'
            ax2.text(j, i, f'{value:.2f}', ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # (c) Sample Grid
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    gs_inner = GridSpec(3, num_experts, figure=fig,
                       left=gs[2].get_position(fig).x0, right=gs[2].get_position(fig).x1,
                       bottom=gs[2].get_position(fig).y0, top=gs[2].get_position(fig).y1,
                       hspace=0.05, wspace=0.05)

    for expert_idx in range(num_experts):
        expert_weights = all_gates[:, expert_idx]
        top_indices = np.argsort(expert_weights)[-100:]
        if len(top_indices) >= 3:
            step = len(top_indices) // 4
            selected = [top_indices[i * step] for i in range(1, 4)]
        else:
            selected = top_indices[-3:].tolist()

        for row, idx in enumerate(selected):
            ax_img = fig.add_subplot(gs_inner[row, expert_idx])
            img = all_images[idx].transpose(1, 2, 0)
            if img.shape[2] == 1:
                ax_img.imshow(img.squeeze(-1), cmap='gray')
            else:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax_img.imshow(img)
            ax_img.axis('off')
            if row == 0:
                ax_img.set_title(expert_names[expert_idx], fontsize=10, fontweight='bold', pad=3)

    fig.suptitle(f'{dataset_name} - {method_name.upper()}', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  Saved: {save_path}")

    return mean_gates, class_gates


def plot_gating_comparison(all_results, dataset, save_path):
    """Compare all 5 methods for one dataset"""
    methods = ['simclr', 'mocov3', 'byol', 'dino', 'barlow']
    expert_names = ['Vis', 'Topo', 'Concat', 'Gated', 'Cross']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'{DATASET_INFO[dataset]["name"]} - Expert Usage Comparison',
                fontsize=16, fontweight='bold')

    for col, method in enumerate(methods):
        if method not in all_results[dataset]:
            continue

        mean_gates = all_results[dataset][method]['mean_gates']
        class_gates = all_results[dataset][method]['class_gates']

        ax_bar = axes[0, col]
        ax_bar.bar(range(5), mean_gates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax_bar.set_ylim([0, 0.5])
        ax_bar.set_xticks(range(5))
        ax_bar.set_xticklabels(expert_names, rotation=45, ha='right', fontsize=8)
        ax_bar.set_title(method.upper(), fontsize=12, fontweight='bold')
        ax_bar.grid(axis='y', alpha=0.3)
        if col == 0:
            ax_bar.set_ylabel('Weight', fontsize=11, fontweight='bold')

        ax_heat = axes[1, col]
        im = ax_heat.imshow(class_gates, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.4)
        ax_heat.set_xticks(range(5))
        ax_heat.set_xticklabels(expert_names, rotation=45, ha='right', fontsize=8)
        if col == 0:
            ax_heat.set_yticks(range(class_gates.shape[0]))
            ax_heat.set_yticklabels([f'C{i}' for i in range(class_gates.shape[0])], fontsize=8)
        else:
            ax_heat.set_yticks([])

    fig.colorbar(im, ax=axes[1, :], orientation='horizontal', fraction=0.05, pad=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  Saved comparison: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='stage3_results')
    parser.add_argument('--roi_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='figures/expert_gating')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['isic2019', 'kvasir', 'octmnist', 'organsmnist', 'pathmnist'])
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['simclr', 'mocov3', 'byol', 'dino', 'barlow'])
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--n_h0', type=int, default=48)
    parser.add_argument('--n_h1', type=int, default=96)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['individual', 'comparisons', 'main_paper', 'supplementary']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    print(f"\n{'='*80}")
    print("Expert Gating Analysis")
    print('='*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Methods: {', '.join(args.methods)}")
    print('='*80 + "\n")

    all_results = {d: {} for d in args.datasets}

    for dataset in args.datasets:
        print(f"\nProcessing: {DATASET_INFO[dataset]['name']}")

        imgs_test, labels_test, rois_test, _ = load_dataset_with_rois(
            dataset, 'test', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=42
        )

        augmenter = TopologyAwareAugmentation()
        test_dataset = Stage3TopoImageDataset(
            imgs_test, labels_test, rois_test, augmenter,
            n_h0=args.n_h0, n_h1=args.n_h1
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        class_names = DATASET_INFO[dataset]['classes']

        for method in args.methods:
            print(f"  Analyzing {method.upper()}...")
            checkpoint_path = os.path.join(args.results_dir, dataset, method, 'best_model.pt')
            encoder, epoch, acc = load_model_and_encoder(
                checkpoint_path, device, hidden_dim=args.hidden_dim,
                num_heads=args.num_heads, n_h0=args.n_h0, n_h1=args.n_h1
            )
            if encoder is None:
                continue

            all_gates, all_labels, all_images = collect_gating_data(encoder, test_loader, device)
            save_path = os.path.join(args.output_dir, 'individual', f'{dataset}_{method}_gating.pdf')
            mean_gates, class_gates = plot_gating_analysis_single(
                all_gates, all_labels, all_images,
                DATASET_INFO[dataset]['name'], method, class_names, save_path
            )
            all_results[dataset][method] = {
                'mean_gates': mean_gates, 'class_gates': class_gates,
                'accuracy': acc, 'epoch': epoch
            }

        if all_results[dataset]:
            comp_path = os.path.join(args.output_dir, 'comparisons', f'{dataset}_all_methods.pdf')
            plot_gating_comparison(all_results, dataset, comp_path)

    # Save summary
    summary = {}
    for dataset in args.datasets:
        summary[dataset] = {}
        for method in all_results[dataset]:
            summary[dataset][method] = {
                'mean_gates': all_results[dataset][method]['mean_gates'].tolist(),
                'accuracy': float(all_results[dataset][method]['accuracy']),
            }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
