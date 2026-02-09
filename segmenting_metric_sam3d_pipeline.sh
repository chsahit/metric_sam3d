#!/usr/bin/env bash
#
# Auto-segmentation pipeline that:
# 0. Uses ChatGPT to identify objects in the scene
# 1. Uses GroundedSAM to generate masks automatically
# 2. Runs the standard metric_sam3d pipeline (mesh generation -> scaling -> registration)
#
# Input: capture_folder with only rgb.png, depth.png, intrinsics.npy
# Output: registered OBJ meshes in output_folder/prepared_data/registered_meshes/
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
    echo "  --device, -d   CUDA device to use (default: 1)"
    echo ""
    echo "Required files in capture_folder:"
    echo "  - rgb.png"
    echo "  - depth.png"
    echo "  - intrinsics.npy"
    echo ""
    echo "Required environment variables:"
    echo "  - OPENAI_API_KEY: OpenAI API key for ChatGPT prompting"
    exit 1
fi

# Verify required input files
if [ ! -f "${CAPTURE_FOLDER}/rgb.png" ]; then
    echo "Error: ${CAPTURE_FOLDER}/rgb.png not found"
    exit 1
fi

if [ ! -f "${CAPTURE_FOLDER}/depth.png" ]; then
    echo "Error: ${CAPTURE_FOLDER}/depth.png not found"
    exit 1
fi

if [ ! -f "${CAPTURE_FOLDER}/intrinsics.npy" ]; then
    echo "Error: ${CAPTURE_FOLDER}/intrinsics.npy not found"
    exit 1
fi

# Check for OpenAI API key
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Automatic Segmentation Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Capture folder: ${CAPTURE_FOLDER}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Device: ${DEVICE}"
echo ""

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

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCENECOMPLETE_DIR="${SCRIPT_DIR}/SceneComplete"
PROMPTING_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/prompting/generate_scene_prompts.py"
SEGMENTATION_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/segmentation/segment_objects.py"

# Temporary directory for segmentation outputs
SEGMENTATION_OUTPUT="${OUTPUT_FOLDER}/segmentation_temp"
mkdir -p "${SEGMENTATION_OUTPUT}"

# Prompts file
PROMPTS_FILE="${SEGMENTATION_OUTPUT}/prompts.txt"
PROMPT_MASK_MAPPING="${SEGMENTATION_OUTPUT}/prompt_mask_mapping.txt"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Step 0: Generate Object Prompts (ChatGPT)${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

unset NVCC_PREPEND_FLAGS

conda activate scenecomplete || {
    echo -e "${RED}Error: Could not activate scenecomplete environment${NC}"
    echo "Please create it first with: bash setup_scenecomplete_env.sh"
    exit 1
}

echo "Using ChatGPT to identify objects in the scene..."
python "${PROMPTING_SCRIPT}" \
    --image_path "${CAPTURE_FOLDER}/rgb.png" \
    --output_filepath "${PROMPTS_FILE}" \
    --api_key "${OPENAI_API_KEY}" \
    --model "gpt-4o"

if [ ! -f "${PROMPTS_FILE}" ]; then
    echo -e "${RED}Error: Failed to generate prompts${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Generated prompts:${NC}"
cat "${PROMPTS_FILE}"
echo ""

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Step 1: Segment Objects (GroundedSAM)${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

export CUDA_VISIBLE_DEVICES=${DEVICE}
export PYOPENGL_PLATFORM=egl

echo "Using GroundingDINO + SAM to generate masks..."
python "${SEGMENTATION_SCRIPT}" \
    --image_path "${CAPTURE_FOLDER}/rgb.png" \
    --depth_path "${CAPTURE_FOLDER}/depth.png" \
    --prompts_filepath "${PROMPTS_FILE}" \
    --prompt_mask_mapping_filepath "${PROMPT_MASK_MAPPING}" \
    --save_dirpath "${SEGMENTATION_OUTPUT}/"

# Count generated masks
NUM_MASKS=$(find "${SEGMENTATION_OUTPUT}" -name "*_object_mask.png" | wc -l)
echo ""
echo -e "${GREEN}Generated ${NUM_MASKS} object masks${NC}"

if [ ${NUM_MASKS} -eq 0 ]; then
    echo -e "${RED}Error: No masks were generated${NC}"
    exit 1
fi

# Copy masks to the expected location (capture_folder/masks/)
echo ""
echo "Copying masks to ${CAPTURE_FOLDER}/masks/..."
mkdir -p "${CAPTURE_FOLDER}/masks"

# Copy {index}_object_mask.png -> masks/{index}.png
for mask_file in "${SEGMENTATION_OUTPUT}"/*_object_mask.png; do
    if [ -f "${mask_file}" ]; then
        # Extract the index (e.g., "0" from "0_object_mask.png")
        basename_file=$(basename "${mask_file}")
        index="${basename_file%%_*}"
        cp "${mask_file}" "${CAPTURE_FOLDER}/masks/${index}.png"
        echo "  Copied ${basename_file} -> masks/${index}.png"
    fi
done

echo -e "${GREEN}Mask generation complete!${NC}"
echo ""

# Now run the standard pipeline
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Standard Metric SAM3D Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${BLUE}===============${NC}"
echo -e "${BLUE}Generate Meshes${NC}"
echo -e "${BLUE}===============${NC}"

# Unset CUDA_VISIBLE_DEVICES for mesh generation (let it access device directly)
unset CUDA_VISIBLE_DEVICES

conda activate sam3d-objects
python generate_meshes.py --capture_folder "${CAPTURE_FOLDER}" --output_folder "${OUTPUT_FOLDER}" --device "${DEVICE}"
conda deactivate

echo -e "${BLUE}========================${NC}"
echo -e "${BLUE}Scaling and Registration${NC}"
echo -e "${BLUE}========================${NC}"

# Paths
PREPARED_DATA_DIR="${OUTPUT_FOLDER}/prepared_data"
GRASP_DATA_DIR="${PREPARED_DATA_DIR}/grasp_data"
IMESH_OUTPUTS="${PREPARED_DATA_DIR}/imesh_outputs"
SCALE_MAPPING="${PREPARED_DATA_DIR}/obj_scale_mapping.txt"
REGISTERED_MESHES="${PREPARED_DATA_DIR}/registered_meshes"

# SceneComplete paths
SCALING_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/scaling/compute_mesh_scaling.py"
REGISTRATION_SCRIPT="${SCENECOMPLETE_DIR}/scenecomplete/scripts/python/registration/register_mesh.py"

unset NVCC_PREPEND_FLAGS

echo ""
echo -e "${GREEN}Step 2: Preparing data for scaling${NC}"

conda activate scenecomplete || {
    echo -e "${RED}Error: Could not activate scenecomplete environment${NC}"
    echo "Please create it first with: bash setup_scenecomplete_env.sh"
    exit 1
}

python prepare_data_for_registration.py --capture_folder "${CAPTURE_FOLDER}" --mesh_folder "${OUTPUT_FOLDER}"

echo ""
echo -e "${GREEN}Step 3: Computing mesh scaling${NC}"

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
echo -e "${GREEN}Step 4: Registering mesh to scene${NC}"
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
echo "  - Segmentation outputs: ${SEGMENTATION_OUTPUT}"
echo "  - Generated prompts: ${PROMPTS_FILE}"
echo "  - Object masks: ${CAPTURE_FOLDER}/masks/"
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
cp "${OUTPUT_FOLDER}/masked_image_"* "${OUTPUT_FOLDER}/results/masks" 2>/dev/null || true
cp "${OUTPUT_FOLDER}/prepared_data/registered_meshes/"* "${OUTPUT_FOLDER}/results/completion_output" 2>/dev/null || true
cp "${OUTPUT_FOLDER}/prepared_data/grasp_data/"[0-9]*_depth.png "${OUTPUT_FOLDER}/results/masks" 2>/dev/null || true

# Optional: zip results
# zip -r "${OUTPUT_FOLDER}/results.zip" "${OUTPUT_FOLDER}/results/"

echo -e "${GREEN}Done! You can visualize results with:${NC}"
echo "  python visualization.py --folder ${OUTPUT_FOLDER}"
