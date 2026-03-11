#!/usr/bin/env python3
"""
Find Failure Cases and Generate Figure 1

Workflow:
1. Load Stage 1 (baseline) and Stage 3 (TopoCL) encoders
2. Train/load linear probes for both models
3. Find failure cases: baseline WRONG, TopoCL CORRECT
4. Save failure case indices to JSON
5. Generate Figure 1: side-by-side Grad-CAM comparison

Usage:
  python analysis/find_failure_cases.py --method dino --dataset isic2019 --stage3_dir ./stage3_results
  python analysis/find_failure_cases.py --method simclr --dataset pathmnist --load_probes
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from topocl.data.datasets import load_dataset_with_rois
from topocl.data.persistence import compute_multiscale_persistence_full_image
from topocl.models.resnet_encoder import ResNetEncoder
from topocl.models.topo_encoder import HierarchicalTopoEncoder
from topocl.models.moe_fusion import MoEFusedEncoder

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
})

CLASS_NAMES = {
    'isic2019': {
        0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK',
        4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC',
    },
    'kvasir': {
        0: 'Esophagitis', 1: 'Polyps', 2: 'Ulcerative Colitis', 3: 'Dyed Lifted Polyps',
        4: 'Dyed Resection Margins', 5: 'Normal Z Line', 6: 'Normal Cecum', 7: 'Normal Pylorus',
    },
    'pathmnist': {i: f'Class {i}' for i in range(9)},
    'octmnist': {i: ['CNV', 'DME', 'DRUSEN', 'NORMAL'][i] for i in range(4)},
    'organsmnist': {i: f'Organ {i}' for i in range(11)},
}


# ============================================================================
# Linear Probe Training
# ============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def compute_pd_single(args_tuple):
    """Compute PH for a single image (module-level for multiprocessing)"""
    img, n_h0, n_h1 = args_tuple

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    pd = compute_multiscale_persistence_full_image(img, n_h0=n_h0, n_h1=n_h1)
    return pd['h0'], pd['h1']


def extract_features(encoder, imgs, rois, n_h0, n_h1, device, batch_size=128):
    """Extract features from encoder"""
    from multiprocessing import Pool, cpu_count

    encoder.eval()

    print(f"  Extracting features from {len(imgs)} images...")

    # Step 1: Compute PH features (parallel)
    args_list = [(img, n_h0, n_h1) for img in imgs]

    n_workers = min(cpu_count() - 1, 8)
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(compute_pd_single, args_list, chunksize=32),
            total=len(imgs),
            desc="  Computing PH",
            leave=False
        ))

    all_h0 = torch.stack([torch.from_numpy(r[0]) for r in results]).float()
    all_h1 = torch.stack([torch.from_numpy(r[1]) for r in results]).float()

    # Step 2: Extract visual features
    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), batch_size), desc="  Extracting features", leave=False):
            batch_imgs = imgs[i:i+batch_size]
            batch_h0 = all_h0[i:i+batch_size].to(device)
            batch_h1 = all_h1[i:i+batch_size].to(device)

            if batch_imgs.dtype == np.uint8:
                batch_imgs = batch_imgs.astype(np.float32) / 255.0

            batch_imgs = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2).float()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            batch_imgs = (batch_imgs - mean) / std
            batch_imgs = batch_imgs.to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                feats, _ = encoder(batch_imgs, batch_h0, batch_h1)

            all_features.append(feats.cpu().float())

    features = torch.cat(all_features, dim=0)
    print(f"  Extracted features: {features.shape}")

    return features


def train_linear_probe(X_train, y_train, X_val, y_val, num_classes, device,
                       epochs=100, lr=0.01, batch_size=256):
    """Train linear classifier on frozen features"""
    classifier = LinearClassifier(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0
    best_state = None

    print(f"  Training linear probe for {epochs} epochs...")

    for epoch in tqdm(range(epochs), desc="  Training probe", leave=False):
        classifier.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val.to(device))
            val_acc = (val_logits.argmax(1) == y_val.to(device)).float().mean().item() * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()

    classifier.load_state_dict(best_state)
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")

    return classifier, best_val_acc


def get_predictions(classifier, features, device):
    """Get predictions from classifier"""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(features.to(device))
        preds = logits.argmax(1).cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return preds, probs


# ============================================================================
# Find Failure Cases
# ============================================================================

def find_failure_cases(baseline_preds, topocl_preds, ground_truth,
                       baseline_probs, topocl_probs, min_confidence=0.6):
    """Find cases where baseline is wrong but TopoCL is correct"""
    failure_cases = []

    for idx in range(len(ground_truth)):
        gt = ground_truth[idx]

        baseline_wrong = (baseline_preds[idx] != gt)
        topocl_correct = (topocl_preds[idx] == gt)
        topocl_confident = (topocl_probs[idx, topocl_preds[idx]] >= min_confidence)

        if baseline_wrong and topocl_correct and topocl_confident:
            failure_cases.append({
                'idx': int(idx),
                'ground_truth': int(gt),
                'baseline_pred': int(baseline_preds[idx]),
                'topocl_pred': int(topocl_preds[idx]),
                'baseline_conf': float(baseline_probs[idx, baseline_preds[idx]]),
                'topocl_conf': float(topocl_probs[idx, topocl_preds[idx]]),
            })

    failure_cases.sort(key=lambda x: x['topocl_conf'], reverse=True)

    return failure_cases


# ============================================================================
# Generate Figure 1
# ============================================================================

def generate_figure1(baseline_model, topocl_model, test_images, failure_cases,
                     class_names, save_path, method_name, n_examples=12, device='cuda'):
    """Generate Figure 1: Side-by-side comparison of baseline vs TopoCL"""
    n_examples = min(n_examples, len(failure_cases))

    if n_examples == 0:
        print("  No failure cases found!")
        return

    fig = plt.figure(figsize=(8, n_examples * 2.0))
    gs = GridSpec(n_examples, 2, figure=fig,
                  wspace=0.08, hspace=0.15,
                  left=0.04, right=0.98,
                  bottom=0.03, top=0.97)

    baseline_model.eval()
    topocl_model.eval()

    baseline_target = [baseline_model.layer4[-1]]
    topocl_target = [topocl_model.layer4[-1]]

    baseline_cam = GradCAM(model=baseline_model, target_layers=baseline_target)
    topocl_cam = GradCAM(model=topocl_model, target_layers=topocl_target)

    print(f"\n  Generating Figure 1 with {n_examples} failure cases...")

    for row, case in enumerate(tqdm(failure_cases[:n_examples], desc="  Generating figure")):
        idx = case['idx']
        img = test_images[idx]
        gt = case['ground_truth']
        baseline_pred = case['baseline_pred']
        topocl_pred = case['topocl_pred']

        gt_name = class_names.get(gt, f'Class {gt}')
        baseline_name = class_names.get(baseline_pred, f'Class {baseline_pred}')
        topocl_name = class_names.get(topocl_pred, f'Class {topocl_pred}')

        if img.max() > 1:
            img_display = img / 255.0
        else:
            img_display = img.copy()

        if img_display.shape[2] == 1:
            img_display = np.repeat(img_display, 3, axis=2)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std

        try:
            baseline_result = baseline_cam(input_tensor=img_tensor, targets=None)[0]
            baseline_overlay = show_cam_on_image(img_display, baseline_result, use_rgb=True)

            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(baseline_overlay)
            ax.axis('off')

            if row == 0:
                ax.set_title(f'{method_name.upper()}\n(Baseline)',
                           fontsize=11, fontweight='bold', pad=6)

            label_text = f'GT: {gt_name}\nPred: {baseline_name} X\nConf: {case["baseline_conf"]:.2f}'
            ax.text(0.02, 0.98, label_text,
                   transform=ax.transAxes, fontsize=8,
                   va='top', ha='left', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='red', linewidth=1.5, alpha=0.9))

            topocl_result = topocl_cam(input_tensor=img_tensor, targets=None)[0]
            topocl_overlay = show_cam_on_image(img_display, topocl_result, use_rgb=True)

            ax = fig.add_subplot(gs[row, 1])
            ax.imshow(topocl_overlay)
            ax.axis('off')

            if row == 0:
                ax.set_title(f'{method_name.upper()}\n+TopoCL',
                           fontsize=11, fontweight='bold', pad=6,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

            label_text = f'GT: {gt_name}\nPred: {topocl_name} OK\nConf: {case["topocl_conf"]:.2f}'
            ax.text(0.02, 0.98, label_text,
                   transform=ax.transAxes, fontsize=8,
                   va='top', ha='left', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='green', linewidth=1.5, alpha=0.9))

        except Exception as e:
            print(f"    Warning: Row {row} failed: {e}")
            continue

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"  Saved Figure 1: {save_path}")
    plt.close()

    del baseline_cam, topocl_cam


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                       choices=['simclr', 'byol', 'dino', 'barlow', 'mocov3'])
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['kvasir', 'pathmnist', 'octmnist', 'organsmnist', 'isic2019'])

    # Data
    parser.add_argument('--roi_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)

    # Architecture (paper defaults)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_h0', type=int, default=48)
    parser.add_argument('--n_h1', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--fusion_embed_dim', type=int, default=512)

    # Checkpoints
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                       help='Path to Stage 1 baseline encoder checkpoint')
    parser.add_argument('--stage3_dir', type=str, default='./stage3_results')

    # Linear probe
    parser.add_argument('--probe_epochs', type=int, default=100)
    parser.add_argument('--load_probes', action='store_true')
    parser.add_argument('--probe_dir', type=str, default='./linear_probes')

    # Figure generation
    parser.add_argument('--n_examples', type=int, default=12)
    parser.add_argument('--min_confidence', type=float, default=0.6)

    # Output
    parser.add_argument('--output_dir', type=str, default='./failure_cases')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"Finding Failure Cases: {args.method.upper()} on {args.dataset}")
    print('='*80)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print('='*80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.probe_dir, exist_ok=True)

    # Load data
    print("Loading test data...")
    imgs_train, labels_train, rois_train, _ = load_dataset_with_rois(
        args.dataset, 'train', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=42
    )
    imgs_val, labels_val, rois_val, _ = load_dataset_with_rois(
        args.dataset, 'val', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=42
    )
    imgs_test, labels_test, rois_test, _ = load_dataset_with_rois(
        args.dataset, 'test', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=42
    )

    imgs_train_full = np.concatenate([imgs_train, imgs_val], axis=0)
    labels_train_full = np.concatenate([labels_train, labels_val], axis=0)
    rois_train_full = rois_train + rois_val

    print(f"  Train: {len(imgs_train_full)} samples")
    print(f"  Test:  {len(imgs_test)} samples\n")

    num_classes = len(np.unique(labels_train_full))
    class_names = CLASS_NAMES.get(args.dataset, {i: f'Class {i}' for i in range(num_classes)})

    # Load Baseline Encoder (Stage 1)
    print("="*80)
    print("Loading Stage 1 Baseline Encoder")
    print("="*80 + "\n")

    baseline_encoder = ResNetEncoder(variant='resnet50').to(device)
    if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
        state_dict = torch.load(args.stage1_checkpoint, map_location=device)
        baseline_encoder.load_state_dict(state_dict)
        print(f"  Loaded from: {args.stage1_checkpoint}")
    else:
        print("  Warning: No Stage 1 checkpoint specified, using random init")
    baseline_encoder.eval()

    # Load TopoCL Encoder (Stage 3)
    print("\n" + "="*80)
    print("Loading Stage 3 TopoCL Encoder")
    print("="*80 + "\n")

    visual_encoder = ResNetEncoder(variant='resnet50')
    topo_encoder = HierarchicalTopoEncoder(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        n_h0=args.n_h0 * 3,
        n_h1=args.n_h1 * 3
    )

    topocl_encoder = MoEFusedEncoder(
        visual_encoder, topo_encoder,
        embed_dim=args.embed_dim,
        out_dim=args.fusion_embed_dim,
        dropout=args.dropout
    ).to(device)

    stage3_path = os.path.join(args.stage3_dir, args.dataset, args.method, 'best_model.pt')
    if not os.path.exists(stage3_path):
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {stage3_path}")

    checkpoint = torch.load(stage3_path, map_location=device)
    topocl_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    topocl_encoder.eval()

    print(f"  Loaded from: {stage3_path}\n")

    # Extract Features or Load
    baseline_probe_path = os.path.join(args.probe_dir, f'{args.dataset}_{args.method}_baseline_probe.pt')
    topocl_probe_path = os.path.join(args.probe_dir, f'{args.dataset}_{args.method}_topocl_probe.pt')
    baseline_features_path = os.path.join(args.probe_dir, f'{args.dataset}_{args.method}_baseline_features.pt')
    topocl_features_path = os.path.join(args.probe_dir, f'{args.dataset}_{args.method}_topocl_features.pt')

    if args.load_probes and os.path.exists(baseline_features_path) and os.path.exists(topocl_features_path):
        print("="*80)
        print("Loading Pre-computed Features")
        print("="*80 + "\n")

        feature_data = torch.load(baseline_features_path)
        X_train_baseline = feature_data['train']
        X_test_baseline = feature_data['test']

        feature_data = torch.load(topocl_features_path)
        X_train_topocl = feature_data['train']
        X_test_topocl = feature_data['test']
    else:
        print("="*80)
        print("Extracting Features")
        print("="*80 + "\n")

        print("Extracting baseline features...")

        class BaselineWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, img, h0, h1):
                feat = self.encoder(img)
                return feat, None

        baseline_wrapper = BaselineWrapper(baseline_encoder).to(device)
        X_train_baseline = extract_features(
            baseline_wrapper, imgs_train_full, rois_train_full,
            args.n_h0, args.n_h1, device
        )
        X_test_baseline = extract_features(
            baseline_wrapper, imgs_test, rois_test,
            args.n_h0, args.n_h1, device
        )

        torch.save({'train': X_train_baseline, 'test': X_test_baseline}, baseline_features_path)

        print("\nExtracting TopoCL features...")
        X_train_topocl = extract_features(
            topocl_encoder, imgs_train_full, rois_train_full,
            args.n_h0, args.n_h1, device
        )
        X_test_topocl = extract_features(
            topocl_encoder, imgs_test, rois_test,
            args.n_h0, args.n_h1, device
        )

        torch.save({'train': X_train_topocl, 'test': X_test_topocl}, topocl_features_path)

    y_train = torch.from_numpy(labels_train_full).long()
    y_test = torch.from_numpy(labels_test).long()

    # Train or Load Linear Probes
    print("="*80)
    print("Linear Probe Training/Loading")
    print("="*80 + "\n")

    if args.load_probes and os.path.exists(baseline_probe_path):
        baseline_classifier = LinearClassifier(X_train_baseline.shape[1], num_classes).to(device)
        baseline_classifier.load_state_dict(torch.load(baseline_probe_path, map_location=device))
        baseline_classifier.eval()
    else:
        baseline_classifier, _ = train_linear_probe(
            X_train_baseline, y_train, X_test_baseline, y_test,
            num_classes, device, epochs=args.probe_epochs
        )
        torch.save(baseline_classifier.state_dict(), baseline_probe_path)

    if args.load_probes and os.path.exists(topocl_probe_path):
        topocl_classifier = LinearClassifier(X_train_topocl.shape[1], num_classes).to(device)
        topocl_classifier.load_state_dict(torch.load(topocl_probe_path, map_location=device))
        topocl_classifier.eval()
    else:
        topocl_classifier, _ = train_linear_probe(
            X_train_topocl, y_train, X_test_topocl, y_test,
            num_classes, device, epochs=args.probe_epochs
        )
        torch.save(topocl_classifier.state_dict(), topocl_probe_path)

    # Get Predictions and Find Failure Cases
    print("\n" + "="*80)
    print("Finding Failure Cases")
    print("="*80 + "\n")

    baseline_preds, baseline_probs = get_predictions(baseline_classifier, X_test_baseline, device)
    topocl_preds, topocl_probs = get_predictions(topocl_classifier, X_test_topocl, device)

    baseline_acc = (baseline_preds == labels_test).mean() * 100
    topocl_acc = (topocl_preds == labels_test).mean() * 100

    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    print(f"  TopoCL accuracy:   {topocl_acc:.2f}%")
    print(f"  Improvement:       +{topocl_acc - baseline_acc:.2f}%\n")

    failure_cases = find_failure_cases(
        baseline_preds, topocl_preds, labels_test,
        baseline_probs, topocl_probs,
        min_confidence=args.min_confidence
    )

    print(f"  Found {len(failure_cases)} failure cases (baseline wrong, TopoCL correct)")

    if len(failure_cases) == 0:
        print("\n  No failure cases found! Try lowering --min_confidence\n")
        return

    failure_json_path = os.path.join(args.output_dir, f'{args.dataset}_{args.method}_failure_cases.json')
    with open(failure_json_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'method': args.method,
            'baseline_accuracy': float(baseline_acc),
            'topocl_accuracy': float(topocl_acc),
            'num_failure_cases': len(failure_cases),
            'failure_cases': failure_cases[:50]
        }, f, indent=2)

    print(f"  Saved failure cases to: {failure_json_path}\n")

    # Generate Figure 1
    print("="*80)
    print("Generating Figure 1")
    print("="*80)

    figure_path = os.path.join(args.output_dir, f'{args.dataset}_{args.method}_figure1.png')

    generate_figure1(
        baseline_encoder, topocl_encoder.visual_encoder,
        imgs_test, failure_cases,
        class_names, figure_path, args.method,
        n_examples=args.n_examples,
        device=device
    )

    print(f"\n{'='*80}")
    print("Complete!")
    print('='*80)
    print(f"\nOutputs:")
    print(f"  - Failure cases JSON: {failure_json_path}")
    print(f"  - Figure 1:           {figure_path}")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
