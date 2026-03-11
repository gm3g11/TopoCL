#!/usr/bin/env python3
"""ResNet encoder with auto-detection of variant (50 or 101)"""

import torch
import torch.nn as nn


def detect_resnet_variant(state_dict):
    """
    Detect ResNet variant from checkpoint keys.

    Returns:
        'resnet50' or 'resnet101'
    """
    # Count layer3 blocks
    layer3_keys = [k for k in state_dict.keys() if k.startswith('layer3.')]
    if not layer3_keys:
        return 'resnet50'  # Default

    # Extract block indices
    block_indices = set()
    for key in layer3_keys:
        parts = key.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            block_indices.add(int(parts[1]))

    max_block = max(block_indices) if block_indices else 0

    # ResNet50: layer3 has blocks 0-5 (6 blocks)
    # ResNet101: layer3 has blocks 0-22 (23 blocks)
    if max_block >= 6:
        return 'resnet101'
    else:
        return 'resnet50'


class ResNetEncoder(nn.Module):
    """ResNet encoder with auto-detection of variant (50 or 101)"""

    def __init__(self, variant='resnet50'):
        super().__init__()
        from torchvision.models import resnet50, resnet101

        if variant == 'resnet101':
            print("    Using ResNet101 architecture")
            resnet = resnet101(weights=None)
        else:
            print("    Using ResNet50 architecture")
            resnet = resnet50(weights=None)

        # Remove FC layer
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.out_dim = 2048  # Both ResNet50 and ResNet101 output 2048
        self.variant = variant

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# Backward compatibility alias
ResNet50Encoder = ResNetEncoder
