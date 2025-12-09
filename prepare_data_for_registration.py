#!/usr/bin/env python3
"""
Prepare data and run scaling/registration pipeline

This script takes the OBJ files from generate_meshes.py and prepares them
for SceneComplete's scaling and registration steps.
Handles multiple objects with numeric IDs (0.obj, 1.obj, etc.)
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
import trimesh
import open3d as o3d
import cv2
import argparse
import glob


def setup_directories(all_paths):
    """Create all necessary directories"""
    print("Setting up directories...")
    for dir_path in [all_paths["grasp_data_dir"], all_paths["meshes_dir"],
                     all_paths["videos_dir"], all_paths["images_dir"]]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  Created output directory structure in {all_paths['output_dir']}")

def convert_intrinsics(all_paths):
    """Convert intrinsics.npy to cam_K.txt and cam_K.json"""
    print("Converting intrinsics...")
    intrinsics = np.load(all_paths["intrinsics_path"])

    # Save as cam_K.txt (space-separated 3x3 matrix)
    cam_k_txt_path = all_paths["grasp_data_dir"] / "cam_K.txt"
    with open(cam_k_txt_path, 'w') as f:
        for row in intrinsics:
            f.write(" ".join(str(val) for val in row) + "\n")
    print(f"  Saved {cam_k_txt_path}")

    # Save as cam_K.json
    cam_k_json_path = all_paths["grasp_data_dir"] / "cam_K.json"
    # Read image to get dimensions
    rgb_img = cv2.imread(str(all_paths["rgb_path"]))
    height, width, _ = rgb_img.shape

    intrinsics_json = {
        "width": width,
        "height": height,
        "intrinsic_matrix": intrinsics.flatten().tolist()
    }
    with open(cam_k_json_path, 'w') as f:
        json.dump(intrinsics_json, f, indent=2)
    print(f"  Saved {cam_k_json_path}")

    return intrinsics, width, height

def prepare_grasp_data(all_paths, object_id, mask_path):
    """Prepare grasp_data directory with scene images and masks for a specific object"""
    print(f"Preparing grasp_data for object {object_id}...")

    # Copy scene full image (only once)
    scene_full_image = all_paths["grasp_data_dir"] / "scene_full_image.png"
    if not scene_full_image.exists():
        shutil.copy(all_paths["rgb_path"], scene_full_image)
        print(f"  Copied scene RGB to grasp_data/")

    # Copy scene full depth (only once)
    scene_full_depth = all_paths["grasp_data_dir"] / "scene_full_depth.png"
    if not scene_full_depth.exists():
        shutil.copy(all_paths["depth_path"], scene_full_depth)
        print(f"  Copied scene depth to grasp_data/")

    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Copy mask as {id}_mask.png
    cv2.imwrite(str(all_paths["grasp_data_dir"] / f"{object_id}_mask.png"), mask)
    print(f"  Saved object mask to grasp_data/{object_id}_mask.png")

    # Create masked RGB ({id}_masked.png)
    rgb = cv2.imread(str(all_paths["rgb_path"]))
    masked_rgb = rgb.copy()
    masked_rgb[mask == 0] = 0  # Black out non-mask regions
    cv2.imwrite(str(all_paths["grasp_data_dir"] / f"{object_id}_masked.png"), masked_rgb)
    print(f"  Saved masked RGB to grasp_data/{object_id}_masked.png")

    # Create masked depth ({id}_depth.png)
    depth = cv2.imread(str(all_paths["depth_path"]), cv2.IMREAD_ANYDEPTH)
    masked_depth = depth.copy()
    masked_depth[mask == 0] = 0  # Zero out non-mask regions
    cv2.imwrite(str(all_paths["grasp_data_dir"] / f"{object_id}_depth.png"), masked_depth)
    print(f"  Saved masked depth to grasp_data/{object_id}_depth.png")

def prepare_mesh_outputs(all_paths, object_id, mesh_obj_path, intrinsics, width, height):
    """Prepare imesh_outputs directory with mesh and renders for a specific object"""
    print(f"Preparing mesh outputs for object {object_id}...")

    # Check if mesh OBJ exists
    if not mesh_obj_path.exists():
        raise FileNotFoundError(
            f"Mesh OBJ not found at {mesh_obj_path}. "
        )

    # Copy mesh to meshes directory as {id}_rgba.obj
    mesh_dst = all_paths["meshes_dir"] / f"{object_id}_rgba.obj"
    shutil.copy(mesh_obj_path, mesh_dst)
    print(f"  Copied mesh to {mesh_dst}")

    # Copy MTL and texture files if they exist
    mesh_dir = mesh_obj_path.parent
    for ext in ['.mtl', '.png']:
        src_pattern = mesh_obj_path.stem + ext
        src_file = mesh_dir / src_pattern
        if src_file.exists():
            dst_file = all_paths["meshes_dir"] / f"{object_id}_rgba{ext}"
            shutil.copy(src_file, dst_file)
            print(f"  Copied {src_file.name} to meshes/")

    # Also check for material_0.png pattern (common in mesh exports)
    material_pattern = mesh_dir / f"material_{object_id}.png"
    if material_pattern.exists():
        shutil.copy(material_pattern, all_paths["meshes_dir"] / f"material_{object_id}.png")
        print(f"  Copied material_{object_id}.png to meshes/")

    # Render mesh to create {id}_rgba.png and depth
    # Check if GLB file exists (prefer GLB for color preservation)
    glb_path = mesh_obj_path.with_suffix('.glb')
    if glb_path.exists():
        print("  Found GLB file, using it for color rendering...")
        render_mesh_source = glb_path
    else:
        print("  Using OBJ file for rendering...")
        render_mesh_source = mesh_dst

    print("  Rendering mesh for scaling correspondences...")
    render_mesh(all_paths, object_id, render_mesh_source, intrinsics, width, height)

    # Create a placeholder in images/ for the scaling script to find
    # (The scaling script uses this directory to determine which objects exist)
    placeholder_path = all_paths["images_dir"] / f"{object_id}_rgba.png"
    shutil.copy(all_paths["rgb_path"], placeholder_path)
    print(f"  Created placeholder in images/ (needed by scaling script)")

def render_mesh(all_paths, object_id, mesh_path, intrinsics, width, height):
    """Render mesh to create RGB and depth images for DINO correspondences"""
    try:
        # Set environment variable for headless rendering with EGL
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        import pyrender
        from PIL import Image

        # Load mesh
        loaded = trimesh.load(mesh_path, force='mesh')

        # Handle both Scene (GLB) and Mesh (OBJ) objects
        if isinstance(loaded, trimesh.Scene):
            # GLB files load as Scene - extract the geometry with colors
            if len(loaded.geometry) > 0:
                # Get the first mesh (or concatenate all)
                meshes = list(loaded.geometry.values())
                if len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    # Concatenate multiple meshes
                    mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError("Scene has no geometry")
        else:
            mesh = loaded

        # Center mesh at origin (like InstantMesh does)
        # Note: We do NOT normalize to unit size to preserve relative scale
        mesh.vertices -= mesh.centroid

        # Create pyrender mesh - from_trimesh automatically handles vertex colors
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        # Setup scene
        scene = pyrender.Scene()
        scene.add(pr_mesh)

        # Setup camera
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.5],  # Camera at z=1.5
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        # Add ambient light to properly show vertex colors
        # Use lower intensity to avoid washing out colors
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        scene.add(light, pose=camera_pose)

        # Add ambient light
        ambient_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        scene.add(ambient_light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(width, height)
        color, depth = renderer.render(scene)

        # Save RGB
        rgb_path = all_paths["videos_dir"] / f"{object_id}_rgba.png"
        Image.fromarray(color).save(rgb_path)
        print(f"    Saved rendered RGB to {rgb_path}")

        # Save depth as 16-bit PNG (in millimeters, like the input depth)
        depth_normalized = (depth * 1000).astype(np.uint16)  # Convert to mm
        depth_path = all_paths["videos_dir"] / f"{object_id}_rgba_depth.png"
        cv2.imwrite(str(depth_path), depth_normalized)
        print(f"    Saved rendered depth to {depth_path}")

        # Save camera intrinsics JSON
        intrinsics_json = {
            "width": width,
            "height": height,
            "intrinsic_matrix": intrinsics.flatten().tolist()
        }
        json_path = all_paths["videos_dir"] / f"{object_id}_rgba.json"
        with open(json_path, 'w') as f:
            json.dump(intrinsics_json, f, indent=2)
        print(f"    Saved camera intrinsics to {json_path}")

        renderer.delete()

    except ImportError:
        print("    WARNING: pyrender not available, creating placeholder renders...")
        # Create placeholder images if pyrender is not available
        # Copy the masked RGB as a placeholder
        shutil.copy(all_paths["grasp_data_dir"] / f"{object_id}_masked.png", all_paths["videos_dir"] / f"{object_id}_rgba.png")
        shutil.copy(all_paths["grasp_data_dir"] / f"{object_id}_depth.png", all_paths["videos_dir"] / f"{object_id}_rgba_depth.png")

        # Save camera intrinsics JSON
        intrinsics_json = {
            "width": width,
            "height": height,
            "intrinsic_matrix": intrinsics.flatten().tolist()
        }
        json_path = all_paths["videos_dir"] / f"{object_id}_rgba.json"
        with open(json_path, 'w') as f:
            json.dump(intrinsics_json, f, indent=2)

def main():
    print("="*60)
    print("Preparing data for scaling and registration")
    print("="*60)

    parser = argparse.ArgumentParser(description="format data for scaling script")
    parser.add_argument("--capture_folder", type=str, required=True)
    parser.add_argument("--mesh_folder", type=str, required=True)
    args = parser.parse_args()

    captures_dir = Path(args.capture_folder)
    mesh_folder = Path(args.mesh_folder)
    output_dir = mesh_folder / "prepared_data"

    # Find all numeric .obj files in the mesh folder
    mesh_files = sorted(mesh_folder.glob("[0-9]*.obj"))
    if not mesh_files:
        raise FileNotFoundError(f"No numeric .obj files found in {mesh_folder}")

    print(f"Found {len(mesh_files)} mesh files to process")

    # Setup common paths
    all_paths = dict()
    all_paths["capture_dir"] = captures_dir
    all_paths["rgb_path"] = captures_dir / "rgb.png"
    all_paths["depth_path"] = captures_dir / "depth.png"
    all_paths["intrinsics_path"] = captures_dir / "intrinsics.npy"
    all_paths["output_dir"] = output_dir
    all_paths["grasp_data_dir"] = output_dir / "grasp_data"
    all_paths["imesh_outputs_dir"] = output_dir / "imesh_outputs/instant-mesh-large"
    all_paths["meshes_dir"] = all_paths["imesh_outputs_dir"] / "meshes"
    all_paths["videos_dir"] = all_paths["imesh_outputs_dir"] / "videos"
    all_paths["images_dir"] = all_paths["imesh_outputs_dir"] / "images"

    # Setup directories
    setup_directories(all_paths)

    # Convert intrinsics (only once)
    intrinsics, width, height = convert_intrinsics(all_paths)

    # Process each mesh
    for mesh_file in mesh_files:
        # Extract object ID from filename (e.g., "0.obj" -> "0")
        object_id = mesh_file.stem

        # Find corresponding mask that was saved with the mesh
        mask_path = mesh_folder / f"mask_{object_id}.png"
        if not mask_path.exists():
            print(f"WARNING: No mask found for object {object_id} at {mask_path}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing object {object_id}")
        print(f"{'='*60}")

        # Prepare grasp_data for this object
        prepare_grasp_data(all_paths, object_id, mask_path)

        # Prepare mesh outputs for this object
        prepare_mesh_outputs(all_paths, object_id, mesh_file, intrinsics, width, height)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nProcessed {len(mesh_files)} objects")
    print(f"Output directory: {all_paths['output_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()
