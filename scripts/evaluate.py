#!/usr/bin/env python3
"""
Standalone Linear Probe Evaluation for Stage 3 Encoders

Usage:
  python evaluate.py --dataset isic2019 --method simclr
  python evaluate.py --dataset isic2019 --method simclr --checkpoint ./stage3_results/isic2019/simclr/best_model.pt
  python evaluate.py --dataset isic2019 --method simclr --n_runs 5
"""

import os
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

from topocl.models.resnet_encoder import ResNetEncoder, detect_resnet_variant
from topocl.models.topo_encoder import HierarchicalTopoEncoder
from topocl.models.moe_fusion import MoEFusedEncoder
from topocl.data.datasets import load_dataset_with_rois
from topocl.utils.training import set_seed, get_num_classes
from topocl.utils.evaluation import extract_features_batched

warnings.filterwarnings('ignore')


class LinearClassifier(nn.Module):
    """Simple linear classifier with optional L2 normalization"""

    def __init__(self, in_dim, num_classes, l2_normalize=False):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=1)
        return self.classifier(x)


def train_linear_classifier(X_train, y_train, X_val, y_val, num_classes,
                            device, epochs=100, lr=0.01, weight_decay=1e-4,
                            batch_size=256, l2_normalize=False, verbose=True):
    """Train linear classifier on frozen features"""
    classifier = LinearClassifier(X_train.shape[1], num_classes, l2_normalize).to(device)

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=0.9,
        weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    pbar = tqdm(range(epochs), desc="  Training", disable=not verbose)

    for epoch in pbar:
        classifier.train()
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

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

        if verbose and epoch % 10 == 0:
            pbar.set_postfix({'val_acc': f'{val_acc:.1f}%', 'best': f'{best_val_acc:.1f}%'})

    classifier.load_state_dict(best_state)
    if verbose:
        print(f"  Training complete! Best val acc: {best_val_acc:.2f}%\n")

    return classifier, best_val_acc


def evaluate_classifier(classifier, X_test, y_test, device):
    """Comprehensive evaluation"""
    classifier.eval()

    with torch.no_grad():
        logits = classifier(X_test.to(device))
        preds = logits.argmax(1).cpu().numpy()

    y_test_np = y_test.cpu().numpy()

    acc = accuracy_score(y_test_np, preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_np, preds, average='macro', zero_division=0
    )

    per_class_acc = []
    for c in range(len(np.unique(y_test_np))):
        mask = y_test_np == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).mean() * 100
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)

    cm = confusion_matrix(y_test_np, preds)

    return {
        'accuracy': acc,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm,
    }


def load_encoder(method, dataset, checkpoint_path, device, args):
    """Load trained encoder from checkpoint"""
    # Detect architecture from stage1 checkpoint if available
    variant = 'resnet50'

    visual_encoder = ResNetEncoder(variant=variant)
    topo_encoder = HierarchicalTopoEncoder(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        n_h0=args.n_h0 * 3,
        n_h1=args.n_h1 * 3
    )

    moe_encoder = MoEFusedEncoder(
        visual_encoder, topo_encoder,
        embed_dim=args.embed_dim,
        out_dim=args.fusion_embed_dim,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'encoder_state_dict' in checkpoint:
        moe_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        probe_acc = checkpoint.get('probe_acc', 'unknown')
        print(f"  Loaded checkpoint from epoch {epoch} (probe acc: {probe_acc})")
    else:
        moe_encoder.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    return moe_encoder


def main():
    parser = argparse.ArgumentParser(description='Linear Probe Evaluation')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kvasir', 'pathmnist', 'octmnist', 'organsmnist', 'isic2019'])
    parser.add_argument('--method', type=str, required=True,
                        choices=['simclr', 'byol', 'dino', 'barlow', 'mocov3'])
    parser.add_argument('--roi_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='./stage3_results')

    # Architecture (should match training)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_h0', type=int, default=48)
    parser.add_argument('--n_h1', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--fusion_embed_dim', type=int, default=512)

    # Linear probe
    parser.add_argument('--probe_epochs', type=int, default=100)
    parser.add_argument('--probe_lr', type=float, default=0.01)
    parser.add_argument('--probe_wd', type=float, default=1e-4)
    parser.add_argument('--probe_batch_size', type=int, default=256)
    parser.add_argument('--l2_normalize', action='store_true')

    # Evaluation
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--extract_batch_size', type=int, default=128)
    parser.add_argument('--quick', action='store_true')

    parser.add_argument('--save_dir', type=str, default='./linear_probe_results')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.quick:
        args.probe_epochs = 50
        args.n_runs = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 80}")
    print(f"Linear Probe Evaluation")
    print(f"{'=' * 80}")
    print(f"Dataset: {args.dataset}, Method: {args.method.upper()}, Device: {device}")
    print(f"{'=' * 80}\n")

    # Find checkpoint
    if args.checkpoint is None:
        checkpoint_path = os.path.join(
            args.results_dir, args.dataset, args.method, 'best_model.pt'
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint

    print(f"Checkpoint: {checkpoint_path}")

    # Load data
    print("\nLoading dataset...")
    roi_kwargs = {}
    if args.roi_dir:
        roi_kwargs['roi_base_dir'] = args.roi_dir
    if args.data_dir:
        roi_kwargs['data_dir'] = args.data_dir

    imgs_train, labels_train, rois_train, _ = load_dataset_with_rois(
        args.dataset, 'train', seed=args.seed, **roi_kwargs
    )
    imgs_val, labels_val, rois_val, _ = load_dataset_with_rois(
        args.dataset, 'val', seed=args.seed, **roi_kwargs
    )
    imgs_test, labels_test, rois_test, _ = load_dataset_with_rois(
        args.dataset, 'test', seed=args.seed, **roi_kwargs
    )

    num_classes = get_num_classes(args.dataset)
    print(f"  Train: {len(imgs_train)}, Val: {len(imgs_val)}, Test: {len(imgs_test)}, Classes: {num_classes}")

    # Load encoder
    print(f"\nLoading encoder...")
    encoder = load_encoder(args.method, args.dataset, checkpoint_path, device, args)
    encoder.eval()

    # Extract features
    print(f"\nExtracting features...")
    X_train = extract_features_batched(
        encoder, imgs_train, rois_train, args.n_h0, args.n_h1, device,
        batch_size=args.extract_batch_size
    )
    X_val = extract_features_batched(
        encoder, imgs_val, rois_val, args.n_h0, args.n_h1, device,
        batch_size=args.extract_batch_size
    )
    X_test = extract_features_batched(
        encoder, imgs_test, rois_test, args.n_h0, args.n_h1, device,
        batch_size=args.extract_batch_size
    )

    y_train = torch.from_numpy(labels_train).long()
    y_val = torch.from_numpy(labels_val).long()
    y_test = torch.from_numpy(labels_test).long()

    # Run evaluation
    all_test_accs = []

    for run in range(args.n_runs):
        print(f"\nRun {run + 1}/{args.n_runs}")
        set_seed(args.seed + run)

        classifier, val_acc = train_linear_classifier(
            X_train, y_train, X_val, y_val, num_classes, device,
            epochs=args.probe_epochs, lr=args.probe_lr,
            weight_decay=args.probe_wd, batch_size=args.probe_batch_size,
            l2_normalize=args.l2_normalize, verbose=(run == 0)
        )

        results = evaluate_classifier(classifier, X_test, y_test, device)
        all_test_accs.append(results['accuracy'])

        if run == 0:
            print(f"\n{'=' * 80}")
            print(f"  Accuracy:  {results['accuracy']:.2f}%")
            print(f"  Precision: {results['precision']:.2f}%")
            print(f"  Recall:    {results['recall']:.2f}%")
            print(f"  F1:        {results['f1']:.2f}%")
            print(f"{'=' * 80}")

    mean_acc = np.mean(all_test_accs)
    std_acc = np.std(all_test_accs)

    print(f"\nFinal: {mean_acc:.2f}% +/- {std_acc:.2f}% ({args.n_runs} runs)")

    # Save results
    save_dir = os.path.join(args.save_dir, args.dataset, args.method)
    os.makedirs(save_dir, exist_ok=True)

    results_dict = {
        'dataset': args.dataset,
        'method': args.method,
        'checkpoint': checkpoint_path,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'all_accuracies': [float(acc) for acc in all_test_accs],
        'n_runs': args.n_runs,
    }

    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to: {save_dir}\n")


if __name__ == '__main__':
    main()
