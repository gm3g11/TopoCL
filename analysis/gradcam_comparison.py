#!/usr/bin/env python3
"""
Generate GradCAM Comparison: Baseline vs TopoCL
Shows how visual encoder attention changes when trained with topology
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from topocl.models import ResNetEncoder, HierarchicalTopoEncoder, MoEFusedEncoder
from topocl.models.resnet_encoder import detect_resnet_variant
from topocl.data.datasets import load_dataset_with_rois

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
})


def load_stage1_baseline(checkpoint_path, device='cuda'):
    """Load Stage 1 baseline (visual-only)"""
    if not os.path.exists(checkpoint_path):
        print(f"  x Checkpoint not found: {checkpoint_path}")
        return None

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    variant = detect_resnet_variant(state_dict)
    encoder = ResNetEncoder(variant=variant).to(device)

    try:
        encoder.load_state_dict(state_dict, strict=False)
        print(f"  + Loaded Stage 1 baseline ({variant})")
        return encoder
    except Exception as e:
        print(f"  x Failed to load: {e}")
        return None


def load_stage3_topocl(stage3_path, baseline_checkpoint_path, device='cuda'):
    """Load Stage 3 TopoCL visual encoder (trained with topology)"""
    # Detect variant from the baseline checkpoint
    state_dict = torch.load(baseline_checkpoint_path, map_location='cpu', weights_only=False)
    variant = detect_resnet_variant(state_dict)

    visual_encoder = ResNetEncoder(variant=variant)
    topo_encoder = HierarchicalTopoEncoder(
        hidden_dim=512, num_heads=8, dropout=0.15,
        n_h0=96*3, n_h1=192*3
    )

    moe_encoder = MoEFusedEncoder(
        visual_encoder, topo_encoder,
        embed_dim=256, out_dim=512, dropout=0.15
    ).to(device)

    if os.path.exists(stage3_path):
        checkpoint = torch.load(stage3_path, map_location=device)
        moe_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"  + Loaded Stage 3 TopoCL ({variant})")
        # Return the visual encoder part (trained with topology)
        return moe_encoder.visual_encoder
    else:
        print(f"  x Checkpoint not found: {stage3_path}")
        return None


def find_interesting_examples(baseline_model, topocl_visual_encoder, test_images, test_labels,
                               n_examples=4, device='cuda'):
    """Find examples where attention differs most"""
    baseline_model.eval()
    topocl_visual_encoder.eval()

    print("  Finding interesting examples...")

    # Target layers - both are visual encoder layer4[-1]
    baseline_target = [baseline_model.layer4[-1]]
    topocl_target = [topocl_visual_encoder.layer4[-1]]

    baseline_cam = GradCAM(model=baseline_model, target_layers=baseline_target)
    topocl_cam = GradCAM(model=topocl_visual_encoder, target_layers=topocl_target)

    differences = []

    n_check = min(500, len(test_images))
    indices = np.random.choice(len(test_images), n_check, replace=False)

    for idx in tqdm(indices, desc="  Computing differences", leave=False):
        img = test_images[idx]
        label = test_labels[idx]

        # Prepare input (just the image for both models)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0

        try:
            # Get attention maps
            baseline_attn = baseline_cam(input_tensor=img_tensor, targets=None)
            topocl_attn = topocl_cam(input_tensor=img_tensor, targets=None)

            # Compute difference
            diff = np.mean((baseline_attn - topocl_attn) ** 2)

            differences.append({
                'idx': idx,
                'diff': diff,
                'label': label
            })
        except Exception as e:
            continue

    if len(differences) == 0:
        print("  x No valid examples!")
        return []

    # Sort by difference
    differences.sort(key=lambda x: x['diff'], reverse=True)

    # Select diverse examples
    selected = []
    selected_labels = set()

    for item in differences:
        if len(selected) >= n_examples:
            break
        if item['label'] not in selected_labels or len(selected) < n_examples // 2:
            selected.append(item)
            selected_labels.add(item['label'])

    while len(selected) < n_examples and len(selected) < len(differences):
        for item in differences:
            if item not in selected:
                selected.append(item)
                break

    diff_values = [f"{s['diff']:.4f}" for s in selected]
    print(f"  + Selected {len(selected)} examples, diff: {diff_values}")

    return selected


def generate_gradcam_figure(baseline_model, topocl_visual_encoder, test_images, test_labels,
                            selected_examples, save_path, method_name):
    """Generate GradCAM comparison figure"""
    n_examples = len(selected_examples)

    if n_examples == 0:
        print("  x No examples to plot!")
        return

    fig = plt.figure(figsize=(7.5, n_examples * 1.8))
    gs = GridSpec(n_examples, 3, figure=fig,
                  wspace=0.15, hspace=0.20,
                  left=0.05, right=0.98,
                  bottom=0.05, top=0.95)

    baseline_model.eval()
    topocl_visual_encoder.eval()
    device = next(baseline_model.parameters()).device

    # Setup GradCAM
    baseline_target = [baseline_model.layer4[-1]]
    topocl_target = [topocl_visual_encoder.layer4[-1]]

    baseline_cam = GradCAM(model=baseline_model, target_layers=baseline_target)
    topocl_cam = GradCAM(model=topocl_visual_encoder, target_layers=topocl_target)

    for row, example in enumerate(selected_examples):
        idx = example['idx']
        img = test_images[idx]
        label = test_labels[idx]

        # Normalize for display
        if img.max() > 1:
            img_display = img / 255.0
        else:
            img_display = img.copy()

        if img_display.shape[2] == 1:
            img_display = np.repeat(img_display, 3, axis=2)

        # Prepare tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0

        try:
            # Generate attention maps
            baseline_attn = baseline_cam(input_tensor=img_tensor, targets=None)[0]
            topocl_attn = topocl_cam(input_tensor=img_tensor, targets=None)[0]

            # Overlay CAMs
            baseline_overlay = show_cam_on_image(img_display, baseline_attn, use_rgb=True)
            topocl_overlay = show_cam_on_image(img_display, topocl_attn, use_rgb=True)

            # Plot original
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.imshow(img_display)
            ax1.axis('off')
            if row == 0:
                ax1.set_title('Original', fontsize=11, fontweight='bold', pad=8)
            ax1.text(0.05, 0.95, f'Class {label}',
                    transform=ax1.transAxes, fontsize=9,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Plot baseline
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.imshow(baseline_overlay)
            ax2.axis('off')
            if row == 0:
                ax2.set_title(f'{method_name.upper()}', fontsize=11, fontweight='bold', pad=8)

            # Plot TopoCL
            ax3 = fig.add_subplot(gs[row, 2])
            ax3.imshow(topocl_overlay)
            ax3.axis('off')
            if row == 0:
                ax3.set_title(f'{method_name.upper()} + TopoCL', fontsize=11, fontweight='bold', pad=8)

        except Exception as e:
            print(f"    Warning: Failed for row {row}: {e}")
            continue

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.05)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)

    print(f"  + Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='GradCAM Comparison: Baseline vs TopoCL')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['simclr', 'mocov3', 'byol', 'dino', 'barlow'])
    parser.add_argument('--dataset', type=str, default='isic2019')
    parser.add_argument('--n_examples', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='figures/gradcam')
    parser.add_argument('--roi_dir', type=str, default='../TopoSSL',
                       help='Directory containing ROI masks')
    parser.add_argument('--stage1_dir', type=str, required=True,
                       help='Directory containing Stage 1 baseline checkpoints (e.g., path/to/{method}_{dataset}.pt)')
    parser.add_argument('--stage3_dir', type=str, default='stage3_results',
                       help='Directory containing Stage 3 results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print("GradCAM Comparison: Baseline vs TopoCL")
    print('='*80)
    print(f"Methods: {', '.join([m.upper() for m in args.methods])}")
    print(f"Dataset: {args.dataset}")
    print(f"Examples per method: {args.n_examples}")
    print(f"Layer: visual_encoder.layer4[-1]")
    print(f"Note: Comparing visual attention patterns")
    print(f"      Baseline: trained visual-only")
    print(f"      TopoCL: trained with topology guidance")
    print('='*80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    imgs_test, labels_test, _, _ = load_dataset_with_rois(
        args.dataset, 'test', roi_base_dir=args.roi_dir, seed=42
    )
    print(f"+ Loaded {len(imgs_test)} test images\n")

    # Process each method
    for method in args.methods:
        print(f"\n{'='*80}")
        print(f"Processing: {method.upper()}")
        print('='*80)

        print("Loading models...")
        baseline_checkpoint = os.path.join(args.stage1_dir, f'{method}_{args.dataset}_encoder.pt')
        stage3_path = os.path.join(args.stage3_dir, args.dataset, method, 'best_model.pt')

        baseline_model = load_stage1_baseline(baseline_checkpoint, device)
        topocl_visual_encoder = load_stage3_topocl(stage3_path, baseline_checkpoint, device)

        if baseline_model is None or topocl_visual_encoder is None:
            print(f"  x Skipping {method}\n")
            continue

        # Find examples
        selected_examples = find_interesting_examples(
            baseline_model, topocl_visual_encoder,
            imgs_test, labels_test,
            n_examples=args.n_examples,
            device=device
        )

        if len(selected_examples) == 0:
            print(f"  x No examples found\n")
            continue

        # Generate figure
        output_path = os.path.join(args.output_dir, f'{method}_{args.dataset}_gradcam.pdf')
        print("  Generating figure...")
        generate_gradcam_figure(
            baseline_model, topocl_visual_encoder,
            imgs_test, labels_test,
            selected_examples, output_path,
            method
        )

    print(f"\n{'='*80}")
    print("+ Complete!")
    print('='*80)
    print(f"\nOutput: {args.output_dir}/")
    print(f"\nInterpretation:")
    print(f"   Baseline: Shows texture/color attention")
    print(f"   TopoCL: Shows boundary/structure attention")
    print(f"   Difference: Impact of topology-guided training")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
