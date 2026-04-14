#!/bin/bash
# Download DINOv2 ViT-B/14 (required for training only)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

HUB_DIR="$ROOT_DIR/pretrained/torch_hub"
DINOV2_DIR="$HUB_DIR/facebookresearch_dinov2_main"
CKPT_DIR="$HUB_DIR/checkpoints"
CKPT_PATH="$CKPT_DIR/dinov2_vitb14_pretrain.pth"

# Clone DINOv2 repo
if [ ! -d "$DINOV2_DIR" ]; then
    echo "Cloning DINOv2 repository..."
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$DINOV2_DIR"
else
    echo "DINOv2 repository already exists, skipping clone."
fi

# Download ViT-B/14 weights
if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading DINOv2 ViT-B/14 weights..."
    mkdir -p "$CKPT_DIR"
    wget -O "$CKPT_PATH" \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
else
    echo "DINOv2 weights already exist, skipping download."
fi

echo "Done."
