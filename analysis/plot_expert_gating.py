#!/usr/bin/env python3
"""
Publication-Ready Expert Gating Figure

Creates a polished 2x5 grid showing expert gating weights:
  Rows: BYOL, MoCo-v3
  Cols: PathMNIST, OCTMNIST, OrganSMNIST, ISIC2019, Kvasir

Usage:
    python analysis/plot_expert_gating.py --stats_json figures/expert_gating/summary.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

DATASET_INFO = {
    'pathmnist': {'name': 'Path'},
    'octmnist': {'name': 'OCT'},
    'organsmnist': {'name': 'OrganS'},
    'isic2019': {'name': 'ISIC'},
    'kvasir': {'name': 'Kvasir'}
}

EXPERT_COLORS = {
    'Vis-Only': '#3498db',
    'Topo-Only': '#e74c3c',
    'Concat': '#2ecc71',
    'Gated': '#f39c12',
    'Cross-Attn': '#9b59b6'
}

EXPERT_NAMES = ['Vis-Only', 'Topo-Only', 'Concat', 'Gated', 'Cross-Attn']
EXPERT_SHORT = ['Vis', 'Topo', 'Conc', 'Gate', 'Attn']


def plot_final_polished(stats_data, save_path):
    """Final polished figure - publication ready"""
    methods = ['byol', 'mocov3']
    datasets = ['pathmnist', 'octmnist', 'organsmnist', 'isic2019', 'kvasir']

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = GridSpec(2, 5, figure=fig,
                  wspace=0.25, hspace=0.20,
                  left=0.10, right=0.98,
                  bottom=0.16, top=0.94)

    max_val = 0
    for dataset in datasets:
        for method in methods:
            if dataset in stats_data and method in stats_data[dataset]:
                data = stats_data[dataset][method]
                mean_gates = np.array(data['mean_gates'])
                std_gates = np.array(data.get('std_gates', [0]*5))
                max_val = max(max_val, (mean_gates + std_gates).max())

    y_max = min(0.5, max_val * 1.25)

    for row, method in enumerate(methods):
        for col, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[row, col])

            if dataset not in stats_data or method not in stats_data[dataset]:
                ax.axis('off')
                continue

            data = stats_data[dataset][method]
            mean_gates = np.array(data['mean_gates'])
            std_gates = np.array(data.get('std_gates', [0]*5))

            x_pos = np.arange(len(EXPERT_NAMES))
            colors = [EXPERT_COLORS[name] for name in EXPERT_NAMES]

            ax.bar(x_pos, mean_gates, yerr=std_gates,
                   color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.0, width=0.65,
                   capsize=3, error_kw={'linewidth': 1.2, 'ecolor': 'black'})

            ax.set_ylim([0, y_max])
            ax.set_xticks(x_pos)

            if row == 1:
                ax.set_xticklabels(EXPERT_SHORT, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel('Weight', fontsize=9, fontweight='bold', labelpad=2)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False)

            ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

            if row == 0:
                ax.set_title(DATASET_INFO[dataset]['name'], fontsize=10, fontweight='bold', pad=5)

            if col == 0:
                method_label = 'BYOL' if method == 'byol' else 'MoCo-v3'
                ax.text(-0.60, 0.5, method_label,
                       transform=ax.transAxes,
                       fontsize=10, fontweight='bold',
                       rotation=90, va='center', ha='center')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.02)
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"\nFinal polished figure saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_json', type=str,
                       default='figures/expert_gating/summary.json')
    parser.add_argument('--output', type=str,
                       default='figures/expert_gating/expert_gating_final.pdf')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Publication-Ready Expert Gating Figure")
    print('='*80 + "\n")

    if not os.path.exists(args.stats_json):
        print(f"Error: JSON not found: {args.stats_json}")
        return

    with open(args.stats_json, 'r') as f:
        stats_data = json.load(f)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_final_polished(stats_data, args.output)

    print(f"\n{'='*80}")
    print("Publication-Ready Figure Complete!")
    print('='*80 + "\n")


if __name__ == '__main__':
    main()
