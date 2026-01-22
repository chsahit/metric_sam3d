# Metric SAM3D Pipeline

Generates metric-scale, pose-registered 3D meshes from RGB-D images and object masks.

## Installation

```bash
git submodule update --init --recursive
# Follow docs for: sam3d-objects, scenecomplete, foundationpose
bash setup_envs_properly.sh
```

If conda is not at `/home/$USER/miniconda3`, update `LD_LIBRARY_PATH` in `metric_sam3d_pipeline.sh`.

## Usage

```bash
./metric_sam3d_pipeline.sh [--device 0] <capture_folder> <output_folder>
```

### Input Structure

```
capture_folder/
├── rgb.png              # RGB image
├── depth.png            # 16-bit PNG, depth in millimeters
├── intrinsics.npy       # 3x3 camera matrix
└── masks/
    └── *.png            # White=object, black=background
```

### Output

Registered meshes: `output_folder/results/completion_output/*.obj`

## API

```bash
# Start server
python metric_sam3d_api.py

# Call (from any machine)
curl -X POST "http://<ip>:8018/metric_sam3d/" \
    -F "capture_zip=@capture.zip" \
    -F "device=0" \
    --output result.zip
```

**ZIP must be flat** (files at root, not nested):
```bash
cd my_capture && zip -r ../capture.zip .
```

## Visualization

```bash
python visualization.py --folder <output_folder>
```

## TODO

- [ ] Compute masks with SAM3
- [ ] Compute masks with GPT + SAM2/3
- [ ] "Cheap" endpoint using built-in pointmap
