#!/usr/bin/env python3
"""
Prepare data and run scaling/registration pipeline

This script takes the PLY file from generate_meshes.py and registers it to the scene
using SceneComplete's scaling and registration steps.
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

def prepare_grasp_data(all_paths):
    """Prepare grasp_data directory with scene images and masks"""
    print("Preparing grasp_data...")

    # Copy scene full image
    shutil.copy(all_paths["rgb_path"], all_paths["grasp_data_dir"] / "scene_full_image.png")
    print(f"  Copied scene RGB to grasp_data/")

    # Copy scene full depth
    shutil.copy(all_paths["depth_path"], all_paths["grasp_data_dir"] / "scene_full_depth.png")
    print(f"  Copied scene depth to grasp_data/")

    # Load mask
    mask = cv2.imread(str(all_paths["mask_path"]), cv2.IMREAD_GRAYSCALE)

    # Copy mask as 0_mask.png
    cv2.imwrite(str(all_paths["grasp_data_dir"] / "0_mask.png"), mask)
    print(f"  Saved object mask to grasp_data/0_mask.png")

    # Create masked RGB (0_masked.png)
    rgb = cv2.imread(str(all_paths["rgb_path"]))
    masked_rgb = rgb.copy()
    masked_rgb[mask == 0] = 0  # Black out non-mask regions
    cv2.imwrite(str(all_paths["grasp_data_dir"] / "0_masked.png"), masked_rgb)
    print(f"  Saved masked RGB to grasp_data/0_masked.png")

    # Create masked depth (0_depth.png)
    depth = cv2.imread(str(all_paths["depth_path"]), cv2.IMREAD_ANYDEPTH)
    masked_depth = depth.copy()
    masked_depth[mask == 0] = 0  # Zero out non-mask regions
    cv2.imwrite(str(all_paths["grasp_data_dir"] / "0_depth.png"), masked_depth)
    print(f"  Saved masked depth to grasp_data/0_depth.png")

def prepare_mesh_outputs(all_paths, intrinsics, width, height):
    """Prepare imesh_outputs directory with mesh and renders"""
    print("Preparing mesh outputs...")

    # Check if mesh OBJ exists
    if not all_paths["mesh_obj_path"].exists():
        raise FileNotFoundError(
            f"Mesh OBJ not found at {all_paths['mesh_obj_path']}. "
        )

    # Copy mesh to meshes directory as 0_rgba.obj
    mesh_dst = all_paths["meshes_dir"] / "0_rgba.obj"
    shutil.copy(all_paths["mesh_obj_path"], mesh_dst)
    print(f"  Copied mesh to {mesh_dst}")

    # Copy MTL and texture files if they exist
    mesh_dir = all_paths["mesh_obj_path"].parent
    for ext in ['.mtl', '.png']:
        src_pattern = all_paths["mesh_obj_path"].stem + ext
        src_file = mesh_dir / src_pattern
        if src_file.exists():
            dst_file = all_paths["meshes_dir"] / f"0_rgba{ext}"
            shutil.copy(src_file, dst_file)
            print(f"  Copied {src_file.name} to meshes/")

    # Also check for material_0.png pattern (common in mesh exports)
    material_pattern = mesh_dir / "material_0.png"
    if material_pattern.exists():
        shutil.copy(material_pattern, all_paths["meshes_dir"] / "material_0.png")
        print(f"  Copied material_0.png to meshes/")

    # Render mesh to create 0_rgba.png and depth
    print("  Rendering mesh for scaling correspondences...")
    render_mesh(all_paths, mesh_dst, intrinsics, width, height)

    # Create a placeholder in images/ for the object
    placeholder_path = all_paths["images_dir"] / "0_rgba.png"
    shutil.copy(all_paths["rgb_path"], placeholder_path)
    print(f"  Created placeholder in images/")

def render_mesh(all_paths, mesh_path, intrinsics, width, height):
    """Render mesh to create RGB and depth images for DINO correspondences"""
    try:
        # Set environment variable for headless rendering with EGL
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        import pyrender
        from PIL import Image

        # Load mesh
        mesh = trimesh.load(mesh_path)

        # Center and scale mesh for better viewing
        mesh.vertices -= mesh.centroid
        max_extent = np.max(mesh.extents)
        mesh.vertices /= max_extent

        # Create pyrender mesh
        pr_mesh = pyrender.Mesh.from_trimesh(mesh)

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

        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(width, height)
        color, depth = renderer.render(scene)

        # Save RGB
        rgb_path = all_paths["videos_dir"] / "0_rgba.png"
        Image.fromarray(color).save(rgb_path)
        print(f"    Saved rendered RGB to {rgb_path}")

        # Save depth as 16-bit PNG (in millimeters, like the input depth)
        depth_normalized = (depth * 1000).astype(np.uint16)  # Convert to mm
        depth_path = all_paths["videos_dir"] / "0_rgba_depth.png"
        cv2.imwrite(str(depth_path), depth_normalized)
        print(f"    Saved rendered depth to {depth_path}")

        # Save camera intrinsics JSON
        intrinsics_json = {
            "width": width,
            "height": height,
            "intrinsic_matrix": intrinsics.flatten().tolist()
        }
        json_path = all_paths["videos_dir"] / "0_rgba.json"
        with open(json_path, 'w') as f:
            json.dump(intrinsics_json, f, indent=2)
        print(f"    Saved camera intrinsics to {json_path}")

        renderer.delete()

    except ImportError:
        print("    WARNING: pyrender not available, creating placeholder renders...")
        # Create placeholder images if pyrender is not available
        # Copy the masked RGB as a placeholder
        shutil.copy(all_paths["grasp_data_dir"] / "0_masked.png", all_paths["videos_dir"] / "0_rgba.png")
        shutil.copy(all_paths["grasp_data_dir"] / "0_depth.png", all_paths["videos_dir"] / "0_rgba_depth.png")

        # Save camera intrinsics JSON
        intrinsics_json = {
            "width": width,
            "height": height,
            "intrinsic_matrix": intrinsics.flatten().tolist()
        }
        json_path = all_paths["videos_dir"] / "0_rgba.json"
        with open(json_path, 'w') as f:
            json.dump(intrinsics_json, f, indent=2)

def main():
    print("="*60)
    print("Preparing data for scaling and registration")
    print("="*60)

    parser = argparse.ArgumentParser(description="format data for scaling script")
    parser.add_argument("--capture_folder", type=str, required=True)
    parser.add_argument("--mesh_folder", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    args = parser.parse_args()
    captures_dir = Path(args.capture_folder)
    output_dir = Path(args.mesh_folder) / "prepared_data"
    all_paths = dict()
    all_paths["capture_dir"] = captures_dir
    all_paths["mesh_obj_path"] = Path(args.mesh_folder) / "completed_rgb.obj"
    all_paths["rgb_path"] = captures_dir / "rgb.png"
    all_paths["depth_path"] = captures_dir / "depth.png"
    all_paths["intrinsics_path"] = captures_dir / "intrinsics.npy"
    all_paths["mask_path"] = Path(args.mask)
    all_paths["output_dir"] = output_dir
    all_paths["grasp_data_dir"] = output_dir / "grasp_data"
    all_paths["imesh_outputs_dir"] = output_dir / "imesh_outputs/instant-mesh-large"
    all_paths["meshes_dir"] = all_paths["imesh_outputs_dir"] / "meshes"
    all_paths["videos_dir"] = all_paths["imesh_outputs_dir"] / "videos"
    all_paths["images_dir"] = all_paths["imesh_outputs_dir"] / "images"

    # Setup directories
    setup_directories(all_paths)

    # Convert intrinsics
    intrinsics, width, height = convert_intrinsics(all_paths)

    # Prepare grasp_data
    prepare_grasp_data(all_paths)

    # Prepare mesh outputs
    prepare_mesh_outputs(all_paths, intrinsics, width, height)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {all_paths['output_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()
