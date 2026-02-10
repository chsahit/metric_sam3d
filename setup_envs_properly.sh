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
pip install fastapi
pip install python-multipart
pip install uvicorn

echo "Installing timm for DINO features..."
pip install timm
pip install pyrender

echo "Installing scenecomplete package..."
pip install -e SceneComplete/

echo "Installing OpenAI client for ChatGPT prompting..."
pip install openai

echo "Installing gdown for downloading weights..."
pip install gdown

echo "Installing Segment Anything (SAM)..."
cd SceneComplete/scenecomplete/modules/GroundedSegmentAnything/segment_anything
pip install -e .
cd -

echo "Installing GroundingDINO..."
cd SceneComplete/scenecomplete/modules/GroundedSegmentAnything/GroundingDINO
pip install -e . --no-build-isolation
cd -

echo "scenecomplete environment setup complete!"
'
