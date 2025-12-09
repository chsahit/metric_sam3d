#!/usr/bin/env bash
#
# This script:
# 1. Activates sam3d-objects environment and runs completion
# 2. Switches to scenecomplete environment and runs scaling
# 3. Switches to foundationpose environment and runs registration
# 4. Outputs a posed OBJ mesh file
#

set -e  # Exit on error

# Parse command-line arguments
DEVICE="1"  # Default device

while [[ $# -gt 0 ]]; do
    case $1 in
        --device|-d)
            DEVICE="$2"
            shift 2
            ;;
        *)
            if [ -z "${CAPTURE_FOLDER}" ]; then
                CAPTURE_FOLDER="${1%/}"
            elif [ -z "${OUTPUT_FOLDER}" ]; then
                OUTPUT_FOLDER="${1%/}"
            else
                echo "Error: Too many positional arguments"
                exit 1
            fi
            shift
            ;;
    esac
done

mkdir -p "${OUTPUT_FOLDER}"

if [ -z "${CAPTURE_FOLDER}" ] || [ -z "${OUTPUT_FOLDER}" ]; then
    echo "Usage: $0 [--device|-d DEVICE] <capture_folder> <output_folder>"
    echo "Example: $0 --device 0 captures/tab2 outputs/test"
    echo "Example: $0 -d 1 captures/tab2 outputs/test"
    echo ""
    echo "Options:"
    echo "  --device, -d   CUDA device to use (default: 0)"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============${NC}"
echo -e "${BLUE}Generate Meshes${NC}"
echo -e "${BLUE}===============${NC}"

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

conda activate sam3d-objects
python generate_meshes.py --capture_folder "${CAPTURE_FOLDER}" --output_folder "${OUTPUT_FOLDER}" --device "${DEVICE}"
conda deactivate

echo -e "${BLUE}========================${NC}"
echo -e "${BLUE}Scaling and Registration${NC}"
echo -e "${BLUE}========================${NC}"

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PREPARED_DATA_DIR="${OUTPUT_FOLDER}/prepared_data"
GRASP_DATA_DIR="${PREPARED_DATA_DIR}/grasp_data"
IMESH_OUTPUTS="${PREPARED_DATA_DIR}/imesh_outputs"
SCALE_MAPPING="${PREPARED_DATA_DIR}/obj_scale_mapping.txt"
REGISTERED_MESHES="${PREPARED_DATA_DIR}/registered_meshes"

# SceneComplete paths
SCENECOMPLETE_DIR="${SCRIPT_DIR}/SceneComplete"
SCALING_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/scaling/compute_mesh_scaling.py"
REGISTRATION_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/registration/register_mesh.py"

unset NVCC_PREPEND_FLAGS

echo ""
echo -e "${GREEN}Step 1: Preparing data for scaling${NC}"

conda activate scenecomplete || {
    echo -e "${RED}Error: Could not activate scenecomplete environment${NC}"
    echo "Please create it first with: bash setup_scenecomplete_env.sh"
    exit 1
}

python prepare_data_for_registration.py --capture_folder "${CAPTURE_FOLDER}" --mesh_folder "${OUTPUT_FOLDER}"

echo ""
echo -e "${GREEN}Step 2: Computing mesh scaling${NC}"

echo "Running scaling computation..."
echo "Using GPU ${DEVICE}..."
export CUDA_VISIBLE_DEVICES=${DEVICE}

python "${SCALING_SCRIPT}" \
    --segmentation_dirpath "${GRASP_DATA_DIR}" \
    --imesh_outputs "${IMESH_OUTPUTS}" \
    --output_filepath "${SCALE_MAPPING}" \
    --instant_mesh_model "instant-mesh-large" \
    --camera_name "realsense"

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
unset NVCC_PREPEND_FLAGS

# Use specified GPU device
echo "Using GPU ${DEVICE}..."
export CUDA_VISIBLE_DEVICES=${DEVICE}

export LD_LIBRARY_PATH=/home/$USER/miniconda3/envs/foundationpose/lib:$LD_LIBRARY_PATH


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
echo "  - Prepared data: ${PREPARED_DATA_DIR}"
echo "  - Scale mapping: ${SCALE_MAPPING}"
echo "  - Registered meshes: ${REGISTERED_MESHES}/"
echo ""
echo "Listing registered meshes:"
ls -lh "${REGISTERED_MESHES}/" 2>/dev/null || echo "  (No registered meshes yet)"
echo ""

mkdir -p "${OUTPUT_FOLDER}/results"
mkdir -p "${OUTPUT_FOLDER}/results/completion_output"
mkdir -p "${OUTPUT_FOLDER}/results/masks"
cp "${OUTPUT_FOLDER}/masked_image_"* "${OUTPUT_FOLDER}/results/masks"
cp "${OUTPUT_FOLDER}/prepared_data/registered_meshes/"* "${OUTPUT_FOLDER}/results/completion_output"
cp "${OUTPUT_FOLDER}/prepared_data/grasp_data/"[0-9]*_depth.png "${OUTPUT_FOLDER}/results/masks"

# zip -r "${OUTPUT_FOLDER}/results.zip" "${OUTPUT_FOLDER}/results/"
