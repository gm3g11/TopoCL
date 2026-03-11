# TopoCL: Topological Contrastive Learning for Medical Image Analysis

**CVPR 2026**

TopoCL is a 3-stage pipeline that combines visual features with topological descriptors (persistent homology) through a Mixture-of-Experts fusion architecture for self-supervised medical image representation learning.

## Pipeline Overview

1. **Stage 1 - Visual Pretraining**: Self-supervised pretraining of a ResNet-50 backbone using 5 SSL methods (SimCLR, MoCo-v3, BYOL, Barlow Twins, DINO)
2. **Stage 2 - Topology Pretraining**: Train a hierarchical topology encoder on persistent homology features extracted from medical images
3. **Stage 3 - MoE Fusion**: Joint fine-tuning combining visual and topological encoders through a 5-expert Mixture-of-Experts gating mechanism

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
TopoCL/
├── topocl/                    # Core library package
│   ├── models/                # ResNet encoder, topology encoder, MoE fusion, SSL heads
│   ├── data/                  # Datasets, augmentations, persistence computation, ROI extraction
│   ├── losses/                # Contrastive loss functions
│   └── utils/                 # Training and evaluation utilities
├── scripts/                   # Training and evaluation scripts
│   ├── stage1_pretrain.py     # Unified visual encoder pretraining (5 SSL methods)
│   ├── stage2_pretrain.py     # Topology encoder pretraining
│   ├── stage3_finetune.py     # Joint MoE fine-tuning
│   └── evaluate.py            # Standalone linear probe evaluation
├── configs/                   # YAML configuration files
├── analysis/                  # Paper figure generation scripts
└── requirements.txt
```

## Quick Start

### Stage 1: Visual Encoder Pretraining

```bash
python scripts/stage1_pretrain.py --method simclr --dataset pathmnist --epochs 200
python scripts/stage1_pretrain.py --method dino --dataset isic2019 --data_dir ~/data/isic2019
```

### Stage 2: Topology Encoder Pretraining

```bash
python scripts/stage2_pretrain.py --dataset pathmnist --method supervised_ce --epochs 100
```

### Stage 3: MoE Fusion Fine-tuning

```bash
python scripts/stage3_finetune.py --dataset pathmnist --epochs 150
```

### Evaluation

```bash
python scripts/evaluate.py --dataset pathmnist --method simclr
```

## Supported Datasets

- **PathMNIST** (9 classes) - Colorectal cancer histology
- **OctMNIST** (4 classes) - Retinal OCT
- **OrganSMNIST** (11 classes) - Abdominal CT organ segmentation
- **ISIC2019** (8 classes) - Skin lesion dermoscopy
- **Kvasir** (8 classes) - Gastrointestinal endoscopy

## Key Architecture Details

- **Topology Encoder**: Hierarchical encoder with separate H0/H1 processing, cross-homology attention (hidden_dim=384, 4 heads)
- **MoE Fusion**: 5 experts (Visual-only, Topology-only, Concatenation, Cross-Attention, Gated Fusion) with meta-gating
- **Persistence Features**: Multi-scale (1.0, 0.5, 0.25) persistent homology with n_h0=48, n_h1=96 per scale

## Citation

```bibtex
@inproceedings{topocl2026,
  title={TopoCL: Topological Contrastive Learning for Medical Image Analysis},
  author={},
  booktitle={CVPR},
  year={2026}
}
```
