#!/usr/bin/env bash
# Properly setup scenecomplete and foundationpose environments using conda run

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
pip install -e /data/sahit/metric_sam3d/SceneComplete

echo "scenecomplete environment setup complete!"
'

echo ""
echo "==========================================="
echo "Setting up foundationpose environment"
echo "==========================================="

# Install PyTorch with CUDA
conda run -n foundationpose pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
conda run -n foundationpose pip install numpy==1.26.4 scipy==1.12.0 scikit-learn==1.4.1.post1
conda run -n foundationpose pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80
conda run -n foundationpose pip install open3d==0.18.0
conda run -n foundationpose pip install trimesh==4.2.2 xatlas==0.0.9
conda run -n foundationpose pip install imageio==2.34.0 matplotlib pillow

# Install PyRender
conda run -n foundationpose pip install pyrender==0.1.45 "pyOpenGL>=3.1.0" "pyOpenGL_accelerate>=3.1.0"

# Install other deps
conda run -n foundationpose pip install omegaconf==2.3.0 roma==1.4.4 einops==0.7.0
conda run -n foundationpose pip install ninja pybind11 transformations

# Install PyTorch3D
conda run -n foundationpose pip install fvcore==0.1.5.post20221221
conda run -n foundationpose pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install nvdiffrast
conda run -n foundationpose pip install git+https://github.com/NVlabs/nvdiffrast/

# Install kaolin
conda run -n foundationpose pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

echo ""
echo "==========================================="
echo "Setup complete!"
echo "==========================================="
echo ""
echo "Verify installations:"
echo "  conda run -n scenecomplete python -c 'import torch; print(torch.__version__)'"
echo "  conda run -n foundationpose python -c 'import torch; print(torch.__version__)'"
