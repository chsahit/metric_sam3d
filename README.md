# Metric SAM3D Pipeline

Generates metric-scale, pose-registered 3D meshes from RGB-D images and object masks.

## Installation

```bash
git submodule update --init --recursive
# Follow docs for: sam3d-objects, scenecomplete, foundationpose
bash setup_envs_properly.sh

# For auto-segmentation pipeline: Set OpenAI API key
export OPENAI_API_KEY="sk-..."  # Add to ~/.bashrc for persistence
```

If conda is not at `/home/$USER/miniconda3`, update `LD_LIBRARY_PATH` in `metric_sam3d_pipeline.sh`.

## Usage

### Standard Pipeline (with pre-computed masks)

```bash
./metric_sam3d_pipeline.sh [--device 0] <capture_folder> <output_folder>
```

**Input Structure:**

```
capture_folder/
├── rgb.png              # RGB image
├── depth.png            # 16-bit PNG, depth in millimeters
├── intrinsics.npy       # 3x3 camera matrix
└── masks/
    └── *.png            # White=object, black=background
```

### Auto-Segmentation Pipeline (masks generated automatically)

Uses ChatGPT to identify objects and GroundedSAM to generate masks automatically:

```bash
./segmenting_metric_sam3d_pipeline.sh [--device 0] <capture_folder> <output_folder>
```

**Input Structure:**

```
capture_folder/
├── rgb.png              # RGB image
├── depth.png            # 16-bit PNG, depth in millimeters
└── intrinsics.npy       # 3x3 camera matrix
# No masks needed - generated automatically!
```

**Requirements:**
- `OPENAI_API_KEY` environment variable must be set
- GroundingDINO and SAM weights (auto-downloaded by `setup_envs_properly.sh`)

**Note:** `SceneComplete/scenecomplete/scripts/python/segmentation/utils/segment_config.yaml` uses absolute paths for GroundingDINO config and weights. If you move the repository or use a different username, update the paths in this file accordingly.

### Output

Registered meshes: `output_folder/results/completion_output/*.obj`

## API

```bash
# Start server (requires OPENAI_API_KEY for /metric_sam3d_full/ endpoint)
python metric_sam3d_api.py
```

### Standard Endpoint (with pre-computed masks)

```bash
# Call (from any machine), takes roughly five minutes
curl -X POST "http://<ip>:8018/metric_sam3d/" \
    -F "capture_zip=@capture.zip" \
    -F "device=0" \
    --output result.zip
```

**Requirements:** ZIP must contain `rgb.png`, `depth.png`, `intrinsics.npy`, and `masks/*.png`

### Auto-Segmentation Endpoint (masks generated automatically)

```bash
# Call (from any machine), takes longer due to auto-segmentation
curl -X POST "http://<ip>:8018/metric_sam3d_full/" \
    -F "capture_zip=@capture.zip" \
    -F "device=0" \
    --output result.zip
```

**Requirements:**
- ZIP must contain only `rgb.png`, `depth.png`, `intrinsics.npy` (no masks needed!)
- Server must have `OPENAI_API_KEY` environment variable set

**Creating the ZIP** (files at root, not nested):
```bash
cd my_capture && zip -r ../capture.zip .
```

## Visualization

```bash
python visualization.py --folder <output_folder>
```

## TODO

- [ ] Compute masks with SAM3
- [x] Compute masks with GPT + SAM2/3 (implemented in `segmenting_metric_sam3d_pipeline.sh`)
- [ ] "Cheap" endpoint using built-in pointmap
