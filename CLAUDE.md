# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Metric SAM3D is a pipeline that generates metric-scale, pose-registered 3D meshes from RGB-D images and object masks. It combines three major components:

1. **sam-3d-objects**: Generates initial 3D meshes from masked images
2. **SceneComplete**: Scales meshes to metric dimensions using DINO-ViT feature correspondences
3. **FoundationPose**: Registers scaled meshes to the scene using 6-DoF pose estimation

The pipeline processes multiple objects in a single scene, outputting posed OBJ files in a common coordinate frame.

## System Architecture

### Pipeline Flow

```
Input (capture_folder/) → generate_meshes.py → prepare_data_for_registration.py → compute_mesh_scaling.py → register_mesh.py → Output (registered OBJs)
```

**Step 1: Mesh Generation (sam3d-objects env)**
- `generate_meshes.py` loads the sam-3d-objects model
- For each mask in `capture_folder/masks/*.png`:
  - Creates masked image using alpha channel
  - Runs SAM 3D Objects inference to generate:
    - `{id}.ply` (Gaussian splat)
    - `{id}.glb` (mesh with texture)
    - `{id}.obj` (mesh for registration)
  - Saves numbered outputs (0.obj, 1.obj, etc.)

**Step 2: Data Preparation (scenecomplete env)**
- `prepare_data_for_registration.py` structures data for SceneComplete modules:
  - Converts intrinsics to `cam_K.txt` and `cam_K.json`
  - Creates `grasp_data/` with scene images and per-object masked RGB/depth
  - Renders each mesh from a fixed viewpoint (radius=4.5, elevation=20°) to create correspondences
  - Organizes into `imesh_outputs/instant-mesh-large/` structure expected by scaling script

**Step 3: Scaling Computation (scenecomplete env)**
- `compute_mesh_scaling.py` computes metric scale for each object:
  - Uses DINO-ViT to find pixel correspondences between rendered mesh and masked scene image
  - Projects correspondences to 3D using depth maps
  - Estimates similarity transform (scale, rotation, translation) via SVD
  - Outputs `obj_scale_mapping.txt` with `{id}:{scale}` per line

**Step 4: Pose Registration (foundationpose env)**
- `register_mesh.py` registers each scaled mesh to the scene:
  - Applies computed scale factor to mesh vertices
  - Uses FoundationPose (differentiable rendering + pose refinement) to find 6-DoF pose
  - Transforms mesh to scene coordinates
  - Exports registered OBJ files to `registered_meshes/`

### Key Data Structures

**Input Structure:**
```
capture_folder/
├── rgb.png              # RGB image
├── depth.png            # 16-bit PNG, depth in millimeters
├── intrinsics.npy       # 3x3 camera intrinsic matrix
└── masks/
    └── *.png            # White=object, black=background
```

**Prepared Data Structure:**
```
prepared_data/
├── grasp_data/
│   ├── cam_K.txt / cam_K.json          # Camera intrinsics
│   ├── scene_full_image.png            # Full scene RGB
│   ├── scene_full_depth.png            # Full scene depth
│   ├── {id}_mask.png                   # Per-object mask
│   ├── {id}_masked.png                 # Per-object masked RGB
│   └── {id}_depth.png                  # Per-object masked depth
├── imesh_outputs/instant-mesh-large/
│   ├── meshes/
│   │   └── {id}_rgba.obj               # Copied mesh
│   ├── videos/
│   │   ├── {id}_rgba.png               # Rendered mesh RGB
│   │   ├── {id}_rgba_depth.png         # Rendered mesh depth
│   │   └── {id}_rgba.json              # Camera intrinsics
│   └── images/
│       └── {id}_rgba.png               # Placeholder for scaling script
├── obj_scale_mapping.txt               # Scale factors: {id}:{scale}
└── registered_meshes/
    └── {id}.obj                        # Final posed mesh
```

### Environment Management

The pipeline uses three conda environments:

**sam3d-objects**: SAM 3D Objects model inference
- PyTorch with CUDA support
- sam3d_objects package
- Used by: `generate_meshes.py`

**scenecomplete**: Scaling and data preparation
- PyTorch (CPU version to avoid conflicts)
- OpenCV, Open3D, pyrender, timm (for DINO)
- scenecomplete package (installed via `pip install -e SceneComplete/`)
- Used by: `prepare_data_for_registration.py`, `compute_mesh_scaling.py`

**foundationpose**: Pose estimation
- FoundationPose dependencies (nvdiffrast, etc.)
- C++ extensions built via `build_all_conda.sh`
- **Important**: Must set `LD_LIBRARY_PATH=/home/$USER/miniconda3/envs/foundationpose/lib` before running
- Used by: `register_mesh.py`

## Common Commands

### Environment Setup
```bash
# Initialize all submodules
git submodule update --init --recursive

# Setup all environments (run from repo root)
bash setup_envs_properly.sh

# If conda is not at /home/$USER/miniconda3, update LD_LIBRARY_PATH in metric_sam3d_pipeline.sh
```

### Running the Pipeline

**Main pipeline (runs all steps):**
```bash
./metric_sam3d_pipeline.sh [--device 0] <capture_folder> <output_folder>
```

**API server:**
```bash
# Start server on port 8018
python metric_sam3d_api.py

# Call from any machine
curl -X POST "http://<ip>:8018/metric_sam3d/" \
    -F "capture_zip=@capture.zip" \
    -F "device=0" \
    --output result.zip

# ZIP must be flat (files at root, not nested)
cd my_capture && zip -r ../capture.zip .
```

**Visualization:**
```bash
# Interactive visualization of registered meshes
python visualization.py --folder <output_folder>

# Modes: 'all' (default), 'colored', 'pcd', 'interactive'
python visualization.py --folder <output_folder> --mode interactive
```

### Running Individual Steps

**Step 1: Generate meshes (sam3d-objects env):**
```bash
conda activate sam3d-objects
python generate_meshes.py \
    --capture_folder captures/tab2 \
    --output_folder outputs/test \
    --device 0
```

**Step 2: Prepare data (scenecomplete env):**
```bash
conda activate scenecomplete
python prepare_data_for_registration.py \
    --capture_folder captures/tab2 \
    --mesh_folder outputs/test
```

**Step 3: Compute scaling (scenecomplete env):**
```bash
conda activate scenecomplete
export CUDA_VISIBLE_DEVICES=0
python SceneComplete/scenecomplete/scripts/python/scaling/compute_mesh_scaling.py \
    --segmentation_dirpath outputs/test/prepared_data/grasp_data \
    --imesh_outputs outputs/test/prepared_data/imesh_outputs \
    --output_filepath outputs/test/prepared_data/obj_scale_mapping.txt \
    --instant_mesh_model "instant-mesh-large" \
    --camera_name "realsense"
```

**Step 4: Register mesh (foundationpose env):**
```bash
conda activate foundationpose
export LD_LIBRARY_PATH=/home/$USER/miniconda3/envs/foundationpose/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
python SceneComplete/scenecomplete/scripts/python/registration/register_mesh.py \
    --imesh_outputs outputs/test/prepared_data/imesh_outputs \
    --segmentation_dirpath outputs/test/prepared_data/grasp_data \
    --obj_scale_mapping outputs/test/prepared_data/obj_scale_mapping.txt \
    --instant_mesh_model "instant-mesh-large" \
    --output_dirpath outputs/test/prepared_data/registered_meshes \
    --est_refine_iter 2 \
    --debug 0
```

## Important Implementation Details

### Coordinate Systems and Transforms

**Camera Conventions:**
- Input depth: 16-bit PNG in millimeters
- Intrinsics: 3x3 matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
- Camera coordinate system: Standard OpenCV (X right, Y down, Z forward)

**Mesh Rendering for Correspondences:**
- Meshes are centered at origin before rendering
- Camera positioned at radius=4.5, elevation=20° (matches SceneComplete paper)
- View direction: looking at origin with up=[0,1,0]
- This consistent viewpoint is critical for DINO correspondence quality

**Scaling via Similarity Transform:**
- DINO finds pixel correspondences between rendered mesh and masked scene image
- Pixels are unprojected to 3D using depth maps and camera intrinsics
- Similarity transform (scale + rotation + translation) estimated via SVD
- Only the scale factor is used (rotation/translation discarded since registration handles pose)

**Registration Coordinate Frame:**
- FoundationPose refines pose using differentiable rendering
- Final transform maps mesh coordinates → scene camera coordinates
- Output OBJ files are in the scene's camera frame (same as input RGB-D)

### Multi-Object Handling

Objects are processed independently with numeric IDs (0, 1, 2, ...):
- Each object has its own mask, scaling factor, and registered pose
- Object ordering is determined by alphabetical sort of mask filenames
- All objects are registered to the same scene coordinate frame
- Final OBJs can be loaded together for scene visualization

### Environment Variable Requirements

**NVCC_PREPEND_FLAGS**: Must be unset when activating scenecomplete or foundationpose envs
```bash
unset NVCC_PREPEND_FLAGS  # Run before conda activate
```

**LD_LIBRARY_PATH**: Required for foundationpose environment
```bash
export LD_LIBRARY_PATH=/home/$USER/miniconda3/envs/foundationpose/lib:$LD_LIBRARY_PATH
```

**PYOPENGL_PLATFORM**: Set to 'egl' for headless rendering
```bash
export PYOPENGL_PLATFORM=egl
```

### GPU Device Selection

All scripts accept a device argument:
```bash
--device 0        # Use CUDA device 0
--device 1        # Use CUDA device 1

# Alternative: Set environment variable
export CUDA_VISIBLE_DEVICES=0
```

## Submodule Details

### sam-3d-objects (git submodule)
- Meta's SAM 3D Objects foundation model
- Generates 3D meshes from single masked images
- Checkpoint location: `sam-3d-objects/checkpoints/hf/`
- Config: `sam-3d-objects/checkpoints/hf/pipeline.yaml`
- Uses InferenceSequential (memory-efficient mode)

### SceneComplete (git submodule)
Contains multiple submodules:

**BrushNet**: Image inpainting for view synthesis
- Located: `SceneComplete/scenecomplete/modules/BrushNet/`
- Not used in current pipeline (pre-generated meshes from SAM 3D)

**GroundedSegmentAnything**: Segmentation (SAM)
- Located: `SceneComplete/scenecomplete/modules/GroundedSegmentAnything/`
- Not used in current pipeline (uses pre-computed masks)

**InstantMesh**: Multi-view reconstruction
- Located: `SceneComplete/scenecomplete/modules/InstantMesh/`
- Not used in current pipeline (SAM 3D Objects used instead)

**FoundationPose**: 6-DoF pose estimation (actively used)
- Located: `SceneComplete/scenecomplete/modules/FoundationPose/`
- C++ extensions: Built via `build_all_conda.sh`
- Models: scorer, refiner (automatically downloaded)

**dino_vit_features**: DINO-ViT for correspondences (actively used)
- Located: `SceneComplete/scenecomplete/modules/dino_vit_features/`
- Used by scaling module for feature matching

### Weights and Models

**SAM 3D Objects weights:**
- Auto-downloaded to `sam-3d-objects/checkpoints/hf/`

**SceneComplete weights:**
- Download script: `SceneComplete/scenecomplete/modules/weights/download_weights.sh`
- Inpainting weights: `modules/weights/inpainting_weights/`
- Pose estimation weights: `modules/weights/pose_estimation_weights/`

## Known Issues and Limitations

1. **Memory Usage**: SAM 3D Objects requires significant GPU memory (>16GB recommended)
   - Uses InferenceSequential mode to reduce memory footprint
   - Process objects one at a time to avoid OOM

2. **FoundationPose LD_LIBRARY_PATH**: Must be set correctly or will get library errors
   - If conda is not at `/home/$USER/miniconda3`, update path in pipeline script

3. **Mask Quality**: Registration accuracy depends on mask quality
   - White pixels = object, black pixels = background
   - Masks should be clean binary images (no anti-aliasing)

4. **Headless Rendering**: Requires EGL support for pyrender
   - Set `PYOPENGL_PLATFORM=egl` for headless servers
   - OSMesa backend is an alternative if EGL unavailable

5. **API Timeout**: Default 30-minute timeout may be insufficient for many objects
   - Adjust timeout in `metric_sam3d_api.py` if needed

## Future TODOs (from README)

- Compute masks with SAM3
- Compute masks with GPT + SAM2/3
- "Cheap" endpoint using built-in pointmap
