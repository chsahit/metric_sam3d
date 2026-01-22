#!/usr/bin/env python3
"""
FastAPI service for metric_sam3d pipeline.

Accepts a zipped capture folder to generate scaled/registered 3D meshes.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
import subprocess
import os
import os.path as osp
import zipfile
import shutil

app = FastAPI(
    title="Metric SAM3D API",
    description="Generate metric-scale 3D meshes from RGB-D images and masks"
)

OUTPUT_DIR = "/home/aditya/research/maggie/metric_sam3d/api_outputs"
PIPELINE_SCRIPT = "/home/aditya/research/maggie/metric_sam3d/metric_sam3d_pipeline.sh"


def zip_folder(folder_to_zip: str, zip_path: str) -> None:
    """Zip the contents of a folder."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                full_file_path = osp.join(root, file)
                arcname = osp.relpath(full_file_path, folder_to_zip)
                zipf.write(full_file_path, arcname=arcname)


def validate_capture_folder(capture_dir: str) -> tuple[bool, str]:
    """Validate that capture folder has required files."""
    required_files = ["rgb.png", "depth.png", "intrinsics.npy"]

    for f in required_files:
        if not osp.exists(osp.join(capture_dir, f)):
            return False, f"Missing required file: {f}"

    masks_dir = osp.join(capture_dir, "masks")
    if not osp.exists(masks_dir):
        return False, "Missing masks/ subfolder"

    mask_count = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    if mask_count == 0:
        return False, "No PNG files found in masks/ subfolder"

    return True, f"Found {mask_count} masks"


@app.post("/metric_sam3d/")
async def metric_sam3d(
    capture_zip: UploadFile = File(..., description="ZIP of capture folder"),
    device: str = Form(default="0", description="CUDA device ID")
):
    """
    Run the metric_sam3d pipeline to generate scaled, registered 3D meshes.

    Input:
    - capture_zip: ZIP file containing:
        - rgb.png: RGB image
        - depth.png: Depth image (16-bit PNG, millimeters)
        - intrinsics.npy: 3x3 camera intrinsic matrix
        - masks/: Folder with object mask PNGs (white=object, black=background)
    - device: CUDA device to use (default: "0")

    Output:
    - ZIP file containing:
        - completion_output/: Registered meshes (.obj) and scene point cloud
        - masks/: Processed masks and depth images
    """
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = osp.join(OUTPUT_DIR, experiment_id)

    capture_dir = osp.join(experiment_dir, "capture")
    output_dir = osp.join(experiment_dir, "output")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save and extract the zip
        zip_path = osp.join(experiment_dir, "capture.zip")
        with open(zip_path, "wb") as f:
            f.write(await capture_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(experiment_dir)

        # Handle both cases: zip contains folder or zip contains files directly
        # Case 1: zip extracts to a subfolder (e.g., capture/rgb.png)
        # Case 2: zip extracts files directly (e.g., rgb.png at root)
        extracted_items = os.listdir(experiment_dir)
        extracted_items = [i for i in extracted_items if i != "capture.zip"]

        if len(extracted_items) == 1 and osp.isdir(osp.join(experiment_dir, extracted_items[0])):
            # Extracted to a single subfolder - rename it to "capture"
            extracted_folder = osp.join(experiment_dir, extracted_items[0])
            if extracted_folder != capture_dir:
                shutil.move(extracted_folder, capture_dir)
        else:
            # Files extracted directly - move them into capture/
            os.makedirs(capture_dir, exist_ok=True)
            for item in extracted_items:
                src = osp.join(experiment_dir, item)
                dst = osp.join(capture_dir, item)
                if src != capture_dir:
                    shutil.move(src, dst)

        # Validate the capture folder
        valid, message = validate_capture_folder(capture_dir)
        if not valid:
            return JSONResponse(
                content={"error": message},
                status_code=400
            )

        # Run the pipeline
        command = [
            "bash",
            PIPELINE_SCRIPT,
            "--device", device,
            capture_dir,
            output_dir
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode != 0:
            return JSONResponse(
                content={
                    "error": "Pipeline failed",
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else ""
                },
                status_code=500
            )

        # Check if results exist
        results_dir = osp.join(output_dir, "results")
        if not osp.exists(results_dir):
            return JSONResponse(
                content={"error": "Pipeline completed but no results directory found"},
                status_code=500
            )

        # Zip the results folder
        result_zip_path = osp.join(experiment_dir, "results.zip")
        zip_folder(results_dir, result_zip_path)

        return FileResponse(
            path=result_zip_path,
            filename=f"metric_sam3d_{experiment_id}.zip",
            media_type='application/zip'
        )

    except subprocess.TimeoutExpired:
        return JSONResponse(
            content={"error": "Pipeline timed out after 30 minutes"},
            status_code=504
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "metric_sam3d"}


if __name__ == "__main__":
    """
    Example usage:

    # Zip your capture folder
    cd /path/to/captures
    zip -r mycapture.zip mycapture/

    # Call the API
    curl -X POST "http://192.168.1.2:8018/metric_sam3d/" \
        -F "capture_zip=@mycapture.zip" \
        -F "device=0" \
        --output result.zip
    """
    import uvicorn

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    uvicorn.run(
        "metric_sam3d_api:app",
        host="0.0.0.0",  # Binds to all interfaces
        port=8018,
        reload=False
    )
