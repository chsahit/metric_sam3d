#!/usr/bin/env bash
# Setup foundationpose with pre-compiled wheels to avoid CUDA version conflicts

set -e

echo "==========================================="
echo "Setting up foundationpose environment"
echo "==========================================="

# Create environment if it doesn't exist
if ! conda env list | grep -q "^foundationpose "; then
    echo "Creating foundationpose environment..."
    conda create -n foundationpose python=3.9 -y
fi

# Run all installations in a single session
conda run -n foundationpose bash -c '
set -e

echo "Installing PyTorch 2.0.1 with CUDA 11.8 (pre-compiled)..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies..."
pip install numpy==1.26.4 scipy==1.12.0 scikit-learn==1.4.1.post1
pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80
pip install open3d==0.18.0
pip install trimesh==4.2.2 xatlas==0.0.9
pip install imageio==2.34.0 matplotlib pillow

echo "Installing PyRender..."
pip install pyrender==0.1.45 "pyOpenGL>=3.1.0" "pyOpenGL_accelerate>=3.1.0"

echo "Installing other dependencies..."
pip install omegaconf==2.3.0 roma==1.4.4 einops==0.7.0
pip install ninja pybind11 transformations

echo "Installing fvcore..."
pip install fvcore==0.1.5.post20221221

echo "Installing PyTorch3D (pre-compiled wheel for torch 2.0.1+cu118)..."
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

echo "Installing nvdiffrast..."
pip install git+https://github.com/NVlabs/nvdiffrast/

echo "Installing kaolin (pre-compiled for torch 2.0.1+cu118)..."
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html

echo "All packages installed successfully!"
'

echo ""
echo "==========================================="
echo "Setup complete!"
echo "==========================================="
echo ""
echo "Verify installation:"
echo "  conda run -n foundationpose python -c 'import torch; print(\"PyTorch:\", torch.__version__)'"
echo "  conda run -n foundationpose python -c 'import pytorch3d; print(\"PyTorch3D: OK\")'"
echo "  conda run -n foundationpose python -c 'import nvdiffrast.torch; print(\"nvdiffrast: OK\")'"
echo "  conda run -n foundationpose python -c 'import kaolin; print(\"kaolin: OK\")'"
