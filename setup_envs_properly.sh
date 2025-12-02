#!/usr/bin/env bash
# Properly setup scenecomplete scaling

set -e

echo "==========================================="
echo "Setting up scenecomplete environment"
echo "==========================================="

# Install packages into scenecomplete environment in a single session
conda run -n scenecomplete bash -c '
set -e

echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

echo "Installing core dependencies with correct versions..."
pip install "numpy<2" scipy matplotlib pillow
pip install "opencv-python<4.10"
pip install open3d==0.18.0

echo "Installing timm for DINO features..."
pip install timm

echo "Installing scenecomplete package..."
pip install -e SceneComplete/

echo "scenecomplete environment setup complete!"
'
