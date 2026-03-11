#!/usr/bin/env python3
"""
Generate and visualize weak/strong augmentations and their persistence diagrams
for PathMNIST. One image per category (9 categories).

Usage:
    python analysis/visualize_pd.py [--output_dir ./pd_visualization]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import medmnist
from medmnist import INFO
from scipy.ndimage import gaussian_filter, rotate
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
import cv2

from topocl.data.roi_extraction import ImprovedROIExtractor

try:
    import gudhi
    USE_GUDHI = True
    print("Using GUDHI for persistence computation")
except ImportError:
    print("Error: GUDHI not found. Please install: pip install gudhi")
    import sys
    sys.exit(1)


def load_pathmnist_highres():
    """Load PathMNIST dataset at 224x224 resolution"""
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split='train', download=True, size=224)
    return dataset.imgs, dataset.labels


def compute_pd_from_image_gudhi(img, n_h0=48, n_h1=96):
    """Compute persistence diagram from image using GUDHI"""
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    img_small = cv2.resize(img, (56, 56), interpolation=cv2.INTER_LINEAR)
    img_norm = ((img_small - img_small.min()) / (img_small.max() - img_small.min() + 1e-8) * 255).astype(np.uint8)
    img_inverted = (255 - img_norm).astype(np.float64)

    cc = gudhi.CubicalComplex(
        dimensions=img_inverted.shape,
        top_dimensional_cells=img_inverted.flatten()
    )
    cc.compute_persistence()
    persistence_pairs = cc.persistence()

    h0_list, h1_list = [], []
    for dim, (birth, death) in persistence_pairs:
        if death != float('inf'):
            birth_norm = birth / 255.0
            death_norm = death / 255.0
            if dim == 0:
                h0_list.append([birth_norm, death_norm])
            elif dim == 1:
                h1_list.append([birth_norm, death_norm])

    h0_all = np.array(h0_list) if h0_list else np.zeros((0, 2))
    h1_all = np.array(h1_list) if h1_list else np.zeros((0, 2))

    if len(h0_all) > 0:
        persistence_h0 = h0_all[:, 1] - h0_all[:, 0]
        top_idx = np.argsort(persistence_h0)[::-1][:n_h0]
        h0 = h0_all[top_idx]
    else:
        h0 = np.zeros((0, 2))

    if len(h1_all) > 0:
        persistence_h1 = h1_all[:, 1] - h1_all[:, 0]
        top_idx = np.argsort(persistence_h1)[::-1][:n_h1]
        h1 = h1_all[top_idx]
    else:
        h1 = np.zeros((0, 2))

    if len(h0) < n_h0:
        h0 = np.vstack([h0, np.zeros((n_h0 - len(h0), 2))]) if len(h0) > 0 else np.zeros((n_h0, 2))
    if len(h1) < n_h1:
        h1 = np.vstack([h1, np.zeros((n_h1 - len(h1), 2))]) if len(h1) > 0 else np.zeros((n_h1, 2))

    return h0, h1


def boundary_noise(img, roi, sigma=0.03):
    """Add reduced noise at ROI boundaries"""
    boundaries = find_boundaries(roi, mode='thick')
    zone = binary_dilation(boundaries, disk(6))

    noise = np.random.normal(0, sigma * 255, img.shape).astype(np.float32)
    out = img.astype(np.float32)

    out[zone] += 2.0 * noise[zone]
    out[~roi] += 0.3 * noise[~roi]
    roi_interior = roi & ~zone
    out[roi_interior] += 0.5 * noise[roi_interior]

    return np.clip(out, 0, 255).astype(np.uint8)


def local_intensity_shift(img, roi, scale=0.12):
    """Reduced local intensity shifts"""
    out = img.astype(np.float32)
    h, w = roi.shape
    grid_size = 56

    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            local_roi = roi[i:i + grid_size, j:j + grid_size]
            if local_roi.any():
                shift = np.random.uniform(-scale, scale)
                patch = out[i:i + grid_size, j:j + grid_size]
                patch[local_roi] *= (1 + shift)
                out[i:i + grid_size, j:j + grid_size] = patch

    return np.clip(out, 0, 255).astype(np.uint8)


def weak_augment_deterministic(img, seed=42):
    """Weak augmentation: Flip + Gaussian noise"""
    np.random.seed(seed)
    img_aug = np.fliplr(img).copy()
    noise = np.random.normal(0, 0.015 * 255, img_aug.shape).astype(np.float32)
    img_aug = img_aug.astype(np.float32) + noise
    return np.clip(img_aug, 0, 255).astype(np.uint8)


def strong_augment_deterministic(img, roi, seed=123):
    """Strong augmentation: Rotation + reduced noise/shifts"""
    np.random.seed(seed)

    if img.ndim == 3:
        img_aug = np.zeros_like(img)
        for c in range(img.shape[2]):
            img_aug[:, :, c] = rotate(img[:, :, c], angle=5, reshape=False, order=1)
    else:
        img_aug = rotate(img, angle=5, reshape=False, order=1)

    img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)
    roi_rotated = rotate(roi.astype(float), angle=5, reshape=False, order=0) > 0.5

    img_aug = boundary_noise(img_aug, roi_rotated, sigma=0.03)
    img_aug = local_intensity_shift(img_aug, roi_rotated, scale=0.12)

    img_aug = img_aug.astype(np.float32)
    img_aug = img_aug * 1.1 + 8
    return np.clip(img_aug, 0, 255).astype(np.uint8)


def plot_pd_large_font(ax, h0, h1, with_title=True, title=''):
    """Plot persistence diagram with large fonts"""
    h0_pers = h0[:, 1] - h0[:, 0]
    h1_pers = h1[:, 1] - h1[:, 0]
    h0_valid = h0[h0_pers > 0]
    h1_valid = h1[h1_pers > 0]

    marker_size = 200

    if len(h0_valid) > 0:
        ax.scatter(h0_valid[:, 0], h0_valid[:, 1], c='blue', alpha=0.7, s=marker_size,
                   label='H0', edgecolors='darkblue', linewidth=3.0)
    if len(h1_valid) > 0:
        ax.scatter(h1_valid[:, 0], h1_valid[:, 1], c='red', alpha=0.7, s=marker_size,
                   label='H1', edgecolors='darkred', linewidth=3.0)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=5.0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel('Birth', fontsize=80, fontweight='bold')
    ax.set_ylabel('Death', fontsize=80, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=65)

    if with_title:
        ax.set_title(title, fontsize=85, fontweight='bold')

    if len(h0_valid) > 0 or len(h1_valid) > 0:
        ax.legend(fontsize=70, loc='lower right', framealpha=0.9)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=4.0)


def select_one_per_class(imgs, labels, num_classes=9):
    """Select one representative image from each class"""
    selected_imgs, selected_labels, selected_indices = [], [], []

    for class_id in range(num_classes):
        class_mask = (labels == class_id).flatten()
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        class_imgs = imgs[class_indices]
        variances = [img.std() for img in class_imgs]
        median_var = np.median(variances)
        best_idx = class_indices[np.argmin([abs(v - median_var) for v in variances])]

        selected_imgs.append(imgs[best_idx])
        selected_labels.append(class_id)
        selected_indices.append(best_idx)
        print(f"  Class {class_id}: Selected image {best_idx}")

    return selected_imgs, selected_labels, selected_indices


def process_single_image(img, label, idx, output_dir):
    """Process a single image and save results"""
    print(f"\nProcessing class {label}, image {idx}...")

    roi_extractor = ImprovedROIExtractor()
    roi = roi_extractor.extract(img)
    print(f"  ROI coverage: {roi.sum() / roi.size * 100:.1f}%")

    weak_img = weak_augment_deterministic(img, seed=42 + idx)
    strong_img = strong_augment_deterministic(img, roi, seed=123 + idx)

    print("  Computing PDs...")
    weak_h0, weak_h1 = compute_pd_from_image_gudhi(weak_img, n_h0=48, n_h1=96)
    strong_h0, strong_h1 = compute_pd_from_image_gudhi(strong_img, n_h0=48, n_h1=96)

    print(f"  Weak   - H0: {(weak_h0[:, 1] - weak_h0[:, 0] > 0).sum()}/48, "
          f"H1: {(weak_h1[:, 1] - weak_h1[:, 0] > 0).sum()}/96")
    print(f"  Strong - H0: {(strong_h0[:, 1] - strong_h0[:, 0] > 0).sum()}/48, "
          f"H1: {(strong_h1[:, 1] - strong_h1[:, 0] > 0).sum()}/96")

    class_dir = os.path.join(output_dir, f'class_{label}')
    os.makedirs(class_dir, exist_ok=True)

    # Save individual images
    for name, data in [('1_original', img), ('2_weak_aug', weak_img), ('3_strong_aug', strong_img)]:
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(data)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(class_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Save PD plots
    for name, h0_data, h1_data in [('4_weak_pd', weak_h0, weak_h1), ('5_strong_pd', strong_h0, strong_h1)]:
        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot(1, 1, 1)
        plot_pd_large_font(ax, h0_data, h1_data, with_title=False)
        plt.tight_layout()
        plt.savefig(os.path.join(class_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Combined figure
    fig = plt.figure(figsize=(30, 20))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img)
    ax1.set_title(f'Original (Class {label})', fontsize=65, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(weak_img)
    ax2.set_title('Weak Augmentation', fontsize=65, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(strong_img)
    ax3.set_title('Strong Augmentation', fontsize=65, fontweight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 5)
    plot_pd_large_font(ax4, weak_h0, weak_h1, with_title=True,
                       title='Weak Aug PD (Top-48 H0 + Top-96 H1)')

    ax5 = plt.subplot(2, 3, 6)
    plot_pd_large_font(ax5, strong_h0, strong_h1, with_title=True,
                       title='Strong Aug PD (Top-48 H0 + Top-96 H1)')

    plt.tight_layout()
    plt.savefig(os.path.join(class_dir, f'combined_class{label}_idx{idx}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(class_dir, 'pd_data.npz'),
             weak_h0=weak_h0, weak_h1=weak_h1,
             strong_h0=strong_h0, strong_h1=strong_h1,
             image_idx=idx, class_label=label)

    print(f"  Saved to {class_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./pd_visualization')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading PathMNIST dataset at 224x224...")
    imgs, labels = load_pathmnist_highres()
    print(f"Loaded {len(imgs)} images")

    num_classes = 9
    print(f"\nSelecting one image per class ({num_classes} classes)...")
    selected_imgs, selected_labels, selected_indices = select_one_per_class(
        imgs, labels, num_classes
    )

    print(f"\nProcessing {len(selected_imgs)} images (one per class)...")

    for img, label, idx in zip(selected_imgs, selected_labels, selected_indices):
        process_single_image(img, label, idx, args.output_dir)

    print(f"\n{'='*80}")
    print(f"All done! Results saved to: {args.output_dir}/")
    print(f"  Generated {len(selected_imgs)} class folders")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
