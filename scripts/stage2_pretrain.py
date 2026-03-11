#!/usr/bin/env python3
"""
Stage 2: Topology Encoder Pretraining

Trains the HierarchicalTopoEncoder using one of three methods:
  - supervised_ce: Cross-entropy with a linear classifier head
  - simclr_ssl: InfoNCE contrastive loss (no labels)
  - supcon: Supervised contrastive loss

Persistence features are computed on topology-augmented views at train time.
"""

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from topocl.data.augmentations import TopologyAwareAugmentation
from topocl.data.persistence import compute_multiscale_persistence_full_image
from topocl.data.datasets import load_dataset_with_rois
from topocl.models.topo_encoder import HierarchicalTopoEncoder
from topocl.losses.contrastive import info_nce_loss, supcon_loss
from topocl.utils.training import set_seed, get_num_classes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TopologyContrastiveDataset(Dataset):
    """
    For each sample, generates two augmented views, computes PH on each,
    and returns the persistence pairs along with the label.
    """

    def __init__(self, imgs, labels, rois, augmenter, n_h0, n_h1, mode):
        """
        Args:
            imgs: (N, H, W, 3) uint8 numpy array
            labels: (N,) numpy array
            rois: list of boolean masks
            augmenter: TopologyAwareAugmentation instance
            n_h0: number of H0 features per scale
            n_h1: number of H1 features per scale
            mode: one of 'supervised_ce', 'simclr_ssl', 'supcon'
        """
        self.imgs = imgs
        self.labels = labels
        self.rois = rois
        self.augmenter = augmenter
        self.n_h0 = n_h0
        self.n_h1 = n_h1
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def _augment(self, img, roi):
        if np.random.rand() < 0.5:
            return self.augmenter.weak_augment(img, roi)
        else:
            return self.augmenter.strong_augment(img, roi)

    def _compute_ph(self, img):
        pd = compute_multiscale_persistence_full_image(
            img, n_h0=self.n_h0, n_h1=self.n_h1
        )
        h0 = torch.from_numpy(pd['h0']).float()
        h1 = torch.from_numpy(pd['h1']).float()
        return h0, h1

    def __getitem__(self, idx):
        img = self.imgs[idx]
        roi = self.rois[idx]
        label = int(self.labels[idx])

        aug1 = self._augment(img, roi)
        aug2 = self._augment(img, roi)

        h0_1, h1_1 = self._compute_ph(aug1)
        h0_2, h1_2 = self._compute_ph(aug2)

        return h0_1, h1_1, h0_2, h1_2, label


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class SupervisedModel(nn.Module):
    """Wraps a topology encoder with a linear classification head."""

    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, h0, h1):
        features = self.encoder(h0, h1)
        logits = self.classifier(features)
        return logits, features


class ContrastiveModel(nn.Module):
    """Wraps a topology encoder with a projection head, then L2-normalises."""

    def __init__(self, encoder, proj_dim=256):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_dim, encoder.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(encoder.out_dim, proj_dim),
        )

    def forward(self, h0, h1):
        features = self.encoder(h0, h1)
        z = self.projector(features)
        z = F.normalize(z, dim=1)
        return z, features


# ---------------------------------------------------------------------------
# Quick linear probe (used after pretraining)
# ---------------------------------------------------------------------------

def quick_linear_probe(encoder, train_loader, test_loader, num_classes, device, epochs=50):
    """Train a small linear probe on top of frozen encoder features."""
    encoder.eval()

    # Collect features
    def collect_features(loader):
        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                h0_1, h1_1, h0_2, h1_2, labels = batch
                h0_1 = h0_1.to(device)
                h1_1 = h1_1.to(device)
                feats = encoder(h0_1, h1_1)
                all_feats.append(feats.cpu())
                all_labels.append(labels)
        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

    train_feats, train_labels = collect_features(train_loader)
    test_feats, test_labels = collect_features(test_loader)

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device).long()
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device).long()

    classifier = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0
    for ep in range(epochs):
        classifier.train()
        logits = classifier(train_feats)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            preds = classifier(test_feats).argmax(dim=1)
            acc = (preds == test_labels).float().mean().item() * 100
            if acc > best_acc:
                best_acc = acc

    return best_acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, method, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        h0_1, h1_1, h0_2, h1_2, labels = batch
        h0_1 = h0_1.to(device)
        h1_1 = h1_1.to(device)
        h0_2 = h0_2.to(device)
        h1_2 = h1_2.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            if method == 'supervised_ce':
                logits1, _ = model(h0_1, h1_1)
                logits2, _ = model(h0_2, h1_2)
                loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2.0

            elif method == 'simclr_ssl':
                z1, _ = model(h0_1, h1_1)
                z2, _ = model(h0_2, h1_2)
                loss = info_nce_loss(z1, z2, temperature=0.1)

            elif method == 'supcon':
                z1, _ = model(h0_1, h1_1)
                z2, _ = model(h0_2, h1_2)
                z = torch.cat([z1, z2], dim=0)
                l = torch.cat([labels, labels], dim=0)
                loss = supcon_loss(z, l, temperature=0.1)

            else:
                raise ValueError(f"Unknown method: {method}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, method, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in loader:
        h0_1, h1_1, h0_2, h1_2, labels = batch
        h0_1 = h0_1.to(device)
        h1_1 = h1_1.to(device)
        h0_2 = h0_2.to(device)
        h1_2 = h1_2.to(device)
        labels = labels.to(device).long()

        with autocast('cuda', dtype=torch.float16):
            if method == 'supervised_ce':
                logits1, _ = model(h0_1, h1_1)
                logits2, _ = model(h0_2, h1_2)
                loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2.0
                preds = logits1.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            elif method == 'simclr_ssl':
                z1, _ = model(h0_1, h1_1)
                z2, _ = model(h0_2, h1_2)
                loss = info_nce_loss(z1, z2, temperature=0.1)

            elif method == 'supcon':
                z1, _ = model(h0_1, h1_1)
                z2, _ = model(h0_2, h1_2)
                z = torch.cat([z1, z2], dim=0)
                l = torch.cat([labels, labels], dim=0)
                loss = supcon_loss(z, l, temperature=0.1)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    acc = (correct / total * 100) if total > 0 else 0.0
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Topology Encoder Pretraining')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (pathmnist, octmnist, organamnist, organsmnist, isic2019, kvasir)')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--roi_dir', type=str, default=None, help='ROI directory')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--method', type=str, required=True,
                        choices=['supervised_ce', 'simclr_ssl', 'supcon'],
                        help='Training method')
    parser.add_argument('--hidden_dim', type=int, default=384, help='Encoder hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    parser.add_argument('--n_h0', type=int, default=48, help='Number of H0 features per scale')
    parser.add_argument('--n_h1', type=int, default=96, help='Number of H1 features per scale')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = get_num_classes(args.dataset)

    print(f"Stage 2 Pretraining: {args.dataset} | method={args.method}")
    print(f"  hidden_dim={args.hidden_dim}, heads={args.num_heads}, dropout={args.dropout}")
    print(f"  n_h0={args.n_h0}, n_h1={args.n_h1}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  epochs={args.epochs}, seed={args.seed}, device={device}")
    print(f"  num_classes={num_classes}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("\nLoading training data...")
    train_imgs, train_labels, train_rois, train_sam = load_dataset_with_rois(
        args.dataset, 'train', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(f"  Train: {train_imgs.shape}, SAM={train_sam}")

    print("Loading validation data...")
    val_imgs, val_labels, val_rois, val_sam = load_dataset_with_rois(
        args.dataset, 'val', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(f"  Val: {val_imgs.shape}, SAM={val_sam}")

    print("Loading test data...")
    test_imgs, test_labels, test_rois, test_sam = load_dataset_with_rois(
        args.dataset, 'test', roi_base_dir=args.roi_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(f"  Test: {test_imgs.shape}, SAM={test_sam}")

    augmenter = TopologyAwareAugmentation()

    train_dataset = TopologyContrastiveDataset(
        train_imgs, train_labels, train_rois, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1, mode=args.method
    )
    val_dataset = TopologyContrastiveDataset(
        val_imgs, val_labels, val_rois, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1, mode=args.method
    )
    test_dataset = TopologyContrastiveDataset(
        test_imgs, test_labels, test_rois, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1, mode=args.method
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    # PH features have shape (n_h0*3, 2) and (n_h1*3, 2) due to 3 scales
    n_h0_total = args.n_h0 * 3
    n_h1_total = args.n_h1 * 3

    encoder = HierarchicalTopoEncoder(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        n_h0=n_h0_total,
        n_h1=n_h1_total,
    )

    if args.method == 'supervised_ce':
        model = SupervisedModel(encoder, num_classes)
        print(f"\n  SupervisedModel: encoder -> Linear({encoder.out_dim}, {num_classes})")
    else:
        model = ContrastiveModel(encoder, proj_dim=256)
        print(f"\n  ContrastiveModel: encoder -> Projector({encoder.out_dim} -> 256)")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # Optimizer / Scheduler / Scaler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')

    # -----------------------------------------------------------------------
    # Save directory
    # -----------------------------------------------------------------------
    if args.save_dir is None:
        args.save_dir = os.path.join('checkpoints', 'stage2', args.dataset, args.method)
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler, args.method, device)
        val_loss, val_acc = eval_epoch(model, val_loader, args.method, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        if args.method == 'supervised_ce':
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.2f}% | lr={lr_now:.2e} | {elapsed:.1f}s")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'method': args.method,
                'dataset': args.dataset,
                'hidden_dim': args.hidden_dim,
                'num_heads': args.num_heads,
                'dropout': args.dropout,
                'n_h0': args.n_h0,
                'n_h1': args.n_h1,
            }
            save_path = os.path.join(args.save_dir, 'best_encoder.pt')
            torch.save(checkpoint, save_path)
            print(f"    -> Saved best model (val_loss={val_loss:.4f})")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")

    # -----------------------------------------------------------------------
    # Load best encoder and run linear probe
    # -----------------------------------------------------------------------
    print("\nRunning quick linear probe evaluation on best encoder...")
    best_ckpt = torch.load(os.path.join(args.save_dir, 'best_encoder.pt'), map_location=device)
    encoder.load_state_dict(best_ckpt['encoder_state_dict'])
    encoder = encoder.to(device)

    probe_acc = quick_linear_probe(
        encoder, train_loader, test_loader, num_classes, device, epochs=50
    )
    print(f"  Linear probe accuracy: {probe_acc:.2f}%")

    # Save final results
    final_checkpoint = torch.load(os.path.join(args.save_dir, 'best_encoder.pt'), map_location='cpu')
    final_checkpoint['linear_probe_acc'] = probe_acc
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'best_encoder.pt'))

    print(f"\nDone. Checkpoint saved to: {os.path.join(args.save_dir, 'best_encoder.pt')}")


if __name__ == '__main__':
    main()
