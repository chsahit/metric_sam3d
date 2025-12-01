#!/usr/bin/env bash
# scratch2.sh - Run scaling and registration pipeline
#
# This script:
# 1. Activates scenecomplete environment and runs scaling
# 2. Switches to foundationpose environment and runs registration
# 3. Outputs a posed OBJ mesh file
#
# Usage:
#   bash scratch2.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}scratch2.sh - Scaling and Registration${NC}"
echo -e "${BLUE}======================================${NC}"

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${SCRIPT_DIR}/captures/tab2/scratch2_output"
GRASP_DATA_DIR="${OUTPUT_DIR}/grasp_data"
IMESH_OUTPUTS="${OUTPUT_DIR}/imesh_outputs"
SCALE_MAPPING="${OUTPUT_DIR}/obj_scale_mapping.txt"
REGISTERED_MESHES="${OUTPUT_DIR}/registered_meshes"

# SceneComplete paths
SCENECOMPLETE_DIR="${SCRIPT_DIR}/SceneComplete"
SCALING_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/scaling/compute_mesh_scaling.py"
REGISTRATION_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/registration/register_mesh.py"

# Check if data preparation has been done
if [ ! -d "$GRASP_DATA_DIR" ]; then
    echo -e "${RED}Error: Data not prepared. Please run scratch2.py first.${NC}"
    exit 1
fi


echo ""
echo -e "${GREEN}Step 2: Computing mesh scaling${NC}"
echo "Activating scenecomplete environment..."

# Try different conda paths
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo -e "${RED}Warning: Could not find conda.sh. Trying to activate directly...${NC}"
    eval "$(conda shell.bash hook)" || true
fi

conda activate scenecomplete || {
    echo -e "${RED}Error: Could not activate scenecomplete environment${NC}"
    echo "Please create it first with: bash setup_scenecomplete_env.sh"
    exit 1
}


echo "Running scaling computation..."
# python "${SCALING_SCRIPT}" \
#     --segmentation_dirpath "${GRASP_DATA_DIR}" \
#     --imesh_outputs "${IMESH_OUTPUTS}" \
#     --output_filepath "${SCALE_MAPPING}" \
#     --instant_mesh_model "instant-mesh-large" \
#     --camera_name "realsense"

echo -e "${GREEN}Scaling computation complete!${NC}"
echo "Scale mapping saved to: ${SCALE_MAPPING}"
cat "${SCALE_MAPPING}"

echo ""
echo -e "${GREEN}Step 3: Registering mesh to scene${NC}"
echo "Activating foundationpose environment..."

conda activate foundationpose || {
    echo -e "${RED}Error: Could not activate foundationpose environment${NC}"
    echo "Please create it first with: bash setup_foundationpose_env.sh"
    exit 1
}

echo "Running mesh registration..."
python "${REGISTRATION_SCRIPT}" \
    --imesh_outputs "${IMESH_OUTPUTS}" \
    --segmentation_dirpath "${GRASP_DATA_DIR}" \
    --obj_scale_mapping "${SCALE_MAPPING}" \
    --instant_mesh_model "instant-mesh-large" \
    --output_dirpath "${REGISTERED_MESHES}" \
    --est_refine_iter 2 \
    --debug 0

echo -e "${GREEN}Registration complete!${NC}"
echo "Registered mesh saved to: ${REGISTERED_MESHES}"
ls -lh "${REGISTERED_MESHES}"

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Pipeline complete!${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Output files:"
echo "  - Scale mapping: ${SCALE_MAPPING}"
echo "  - Registered mesh: ${REGISTERED_MESHES}/0.obj"
echo ""
