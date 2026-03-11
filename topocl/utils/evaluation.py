#!/usr/bin/env python3
"""Evaluation utilities: linear probe, TTA, feature extraction"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
from sklearn.metrics import accuracy_score

from topocl.data.persistence import compute_multiscale_persistence_full_image


def linear_probe_evaluation(encoder, imgs_train, labels_train, rois_train,
                            imgs_test, labels_test, rois_test,
                            n_h0=288, n_h1=576, num_classes=9,
                            device='cuda', epochs=100, batch_size=512):
    """
    Fast linear probe evaluation with batched feature extraction.
    Computes PH on images BEFORE normalization.
    """
    encoder.eval()

    print(f"    Extracting features (batched, batch_size={batch_size})...")

    def extract_features_inner(imgs, rois, batch_size=512):
        features = []
        n_samples = len(imgs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="    Extracting", leave=False):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                batch_imgs = imgs[start_idx:end_idx]

                batch_h0 = []
                batch_h1 = []
                for img in batch_imgs:
                    if img.dtype != np.uint8:
                        img_uint8 = (img * 255).astype(np.uint8)
                    else:
                        img_uint8 = img

                    pd = compute_multiscale_persistence_full_image(
                        img_uint8, n_h0=n_h0 // 3, n_h1=n_h1 // 3
                    )
                    batch_h0.append(torch.from_numpy(pd['h0']))
                    batch_h1.append(torch.from_numpy(pd['h1']))

                batch_h0 = torch.stack(batch_h0).float()
                batch_h1 = torch.stack(batch_h1).float()

                batch_imgs = batch_imgs.astype(np.float32) / 255.0
                batch_imgs = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2)

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                batch_imgs = (batch_imgs - mean) / std

                batch_imgs = batch_imgs.to(device)
                batch_h0 = batch_h0.to(device)
                batch_h1 = batch_h1.to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feats, _ = encoder(batch_imgs, batch_h0, batch_h1)

                features.append(feats.cpu())

        return torch.cat(features, dim=0)

    train_feats = extract_features_inner(imgs_train, rois_train, batch_size)
    test_feats = extract_features_inner(imgs_test, rois_test, batch_size)

    print(f"    Features: train={train_feats.shape}, test={test_feats.shape}")

    train_labels = torch.from_numpy(labels_train).long()
    test_labels = torch.from_numpy(labels_test).long()

    classifier = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=0
    )

    classifier.train()
    for _ in range(epochs):
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        test_feats = test_feats.to(device)
        test_labels = test_labels.to(device)
        logits = classifier(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100

    return acc


def linear_probe_with_tta(
        encoder, linear_classifier,
        imgs_test, labels_test, rois_test,
        augmenter, n_h0, n_h1, device, n_aug=5
):
    """
    Test-time augmentation for final evaluation.
    Averages features over multiple weak augmentations.
    """
    encoder.eval()
    linear_classifier.eval()

    all_preds = []

    print(f"\nRunning Test-Time Augmentation with {n_aug} augmentations...")

    with torch.no_grad():
        for img, roi in tqdm(zip(imgs_test, rois_test), total=len(imgs_test), desc="TTA"):
            features = []

            for _ in range(n_aug):
                img_aug = augmenter.weak_augment(img, roi)

                if img_aug.dtype != np.uint8:
                    img_aug = (img_aug * 255).astype(np.uint8)

                img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                pd = compute_multiscale_persistence_full_image(img_aug, n_h0=n_h0, n_h1=n_h1)
                h0 = torch.from_numpy(pd['h0']).unsqueeze(0).to(device).float()
                h1 = torch.from_numpy(pd['h1']).unsqueeze(0).to(device).float()

                with autocast('cuda', dtype=torch.float16):
                    feat, _ = encoder(img_tensor, h0, h1)
                    feat = feat.float()

                features.append(feat)

            avg_feat = torch.stack(features).mean(dim=0)
            logits = linear_classifier(avg_feat)
            prob = F.softmax(logits, dim=1)
            all_preds.append(prob.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    predicted = all_preds.argmax(dim=1).numpy()

    if isinstance(labels_test, torch.Tensor):
        labels_test = labels_test.cpu().numpy()

    tta_acc = accuracy_score(labels_test, predicted) * 100

    print(f"\nTTA Accuracy: {tta_acc:.2f}%\n")
    return tta_acc


def train_linear_probe_from_encoder(
        encoder,
        imgs_train, labels_train, rois_train,
        imgs_test, labels_test, rois_test,
        n_h0, n_h1, num_classes, device, epochs=100, lr=1e-3
):
    """
    Train a linear classifier on top of frozen encoder features.
    Returns the trained linear classifier for use with TTA.
    """
    encoder.eval()

    print("    Training linear classifier on frozen features...")

    def extract_features(imgs, rois):
        features = []
        for img, roi in tqdm(zip(imgs, rois), total=len(imgs), desc="    Extracting", leave=False):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pd = compute_multiscale_persistence_full_image(img, n_h0=n_h0 // 3, n_h1=n_h1 // 3)
            h0 = torch.from_numpy(pd['h0']).unsqueeze(0).to(device).float()
            h1 = torch.from_numpy(pd['h1']).unsqueeze(0).to(device).float()

            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    feat, _ = encoder(img_tensor, h0, h1)
                    feat = feat.float()

            features.append(feat.cpu())

        return torch.cat(features, dim=0)

    X_train = extract_features(imgs_train, rois_train)
    y_train = torch.from_numpy(labels_train) if isinstance(labels_train, np.ndarray) else labels_train
    X_test = extract_features(imgs_test, rois_test)
    y_test = torch.from_numpy(labels_test) if isinstance(labels_test, np.ndarray) else labels_test

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    linear = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        linear.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(linear(X_train), y_train)
        loss.backward()
        optimizer.step()

        linear.eval()
        with torch.no_grad():
            acc = (linear(X_test).argmax(1) == y_test).float().mean().item() * 100
            if acc > best_acc:
                best_acc = acc
                best_state = linear.state_dict().copy()

    linear.load_state_dict(best_state)

    print(f"    Linear classifier trained: {best_acc:.2f}%")
    return linear, best_acc


def extract_features_batched(encoder, imgs, rois, n_h0, n_h1, device,
                             batch_size=128, use_amp=True, normalize=True):
    """
    Extract features from encoder with optimized parallel processing.
    Uses multiprocessing for PH computation.
    """
    encoder.eval()
    all_features = []

    n_samples = len(imgs)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"  Extracting features: {n_samples} images, batch_size={batch_size}")

    # Pre-compute PH features
    print(f"  Step 1/2: Computing persistence features (parallel)...")

    from multiprocessing import Pool, cpu_count

    def compute_pd_single(args_tuple):
        img, nh0, nh1 = args_tuple
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        else:
            img_uint8 = img
        pd = compute_multiscale_persistence_full_image(img_uint8, n_h0=nh0, n_h1=nh1)
        return pd['h0'], pd['h1']

    args_list = [(img, n_h0, n_h1) for img in imgs]
    n_workers = min(cpu_count() - 1, 8)

    all_h0 = []
    all_h1 = []

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(compute_pd_single, args_list, chunksize=32),
            total=n_samples,
            desc=f"  PH ({n_workers} workers)",
            leave=False
        ))

    for h0, h1 in results:
        all_h0.append(torch.from_numpy(h0))
        all_h1.append(torch.from_numpy(h1))

    all_h0 = torch.stack(all_h0).float()
    all_h1 = torch.stack(all_h1).float()

    # Extract visual features in batches
    print(f"  Step 2/2: Extracting visual features...")

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="  GPU forward", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_imgs = imgs[start_idx:end_idx]
            batch_h0 = all_h0[start_idx:end_idx].to(device)
            batch_h1 = all_h1[start_idx:end_idx].to(device)

            if batch_imgs.dtype == np.uint8:
                batch_imgs = batch_imgs.astype(np.float32) / 255.0

            batch_imgs = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2).float()

            if normalize:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                batch_imgs = (batch_imgs - mean) / std

            batch_imgs = batch_imgs.to(device)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feats, _ = encoder(batch_imgs, batch_h0, batch_h1)
                feats = feats.float()
            else:
                feats, _ = encoder(batch_imgs, batch_h0, batch_h1)

            all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0)
    print(f"  Extracted features: {features.shape}")
    return features
