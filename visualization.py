#!/usr/bin/env python3
"""
Visualization script for metric_sam3d pipeline outputs.

Visualizes completed meshes and scene point cloud from the output of
metric_sam3d_pipeline.sh.
"""

import os
import argparse
import open3d as o3d
import numpy as np
import trimesh


def load_obj_files(obj_folder_path):
    """
    Load all obj file paths from the specified folder.

    Args:
        obj_folder_path (str): Path to folder containing obj files

    Returns:
        list: List of obj file paths (sorted numerically)
    """
    obj_files = []
    for filename in os.listdir(obj_folder_path):
        if filename.endswith('.obj'):
            obj_path = os.path.join(obj_folder_path, filename)
            obj_files.append(obj_path)

    # Sort numerically by filename (e.g., 0.obj, 1.obj, 2.obj)
    obj_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return obj_files


def visualize_colored_meshes(completion_output_path):
    """
    Visualize obj files as colored meshes using trimesh.

    Args:
        completion_output_path (str): Path to completion_output folder
    """
    obj_files = load_obj_files(completion_output_path)

    if not obj_files:
        print(f"No obj files found in {completion_output_path}")
        return

    scene = trimesh.Scene()
    for obj_file in obj_files:
        obj = trimesh.load(obj_file)
        scene.add_geometry(obj)

    print(f"Loaded {len(obj_files)} meshes")
    scene.show()


def visualize_objs_scene_pcd(completion_output_path):
    """
    Visualize obj files along with the scene ply file using Open3D.

    Args:
        completion_output_path (str): Path to completion_output folder
    """
    obj_files = load_obj_files(completion_output_path)
    obj_meshes = []

    for obj_file in obj_files:
        obj_mesh = o3d.io.read_triangle_mesh(obj_file)
        obj_mesh.compute_vertex_normals()
        obj_meshes.append(obj_mesh)

    scene_ply_path = os.path.join(completion_output_path, "scene_complete.ply")
    if not os.path.exists(scene_ply_path):
        raise FileNotFoundError(f"Scene file not found at {scene_ply_path}")

    scene = o3d.io.read_point_cloud(scene_ply_path)

    print(f"Loaded scene with {len(scene.points)} points")
    print(f"Loaded {len(obj_meshes)} meshes")

    o3d.visualization.draw_geometries([scene] + obj_meshes)


def visualize_scene_with_objects(completion_output_path):
    """
    Visualize multiple obj files along with a scene ply file
    using an interactive Open3D visualizer.

    Args:
        completion_output_path (str): Path to completion_output folder
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    scene_ply_path = os.path.join(completion_output_path, "scene_complete.ply")

    if not os.path.exists(scene_ply_path):
        raise FileNotFoundError(f"Scene file not found at {scene_ply_path}")

    scene = o3d.io.read_point_cloud(scene_ply_path)
    if not scene.has_points():
        raise ValueError(f"No points found in scene file {scene_ply_path}")

    vis.add_geometry(scene)

    opt = vis.get_render_option()
    opt.point_size = 1.0

    obj_files = load_obj_files(completion_output_path)
    for filename in obj_files:
        mesh = o3d.io.read_triangle_mesh(filename)
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    opt.show_coordinate_frame = True

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    print(f"Loaded scene with {len(scene.points)} points")
    print(f"Loaded {len(obj_files)} meshes")

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize metric_sam3d pipeline outputs'
    )
    parser.add_argument(
        '--folder', '-f',
        type=str,
        required=True,
        help='Path to pipeline output folder (e.g., test0_metric)'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['all', 'colored', 'pcd', 'interactive'],
        default='all',
        help='Visualization mode: all (run all), colored (trimesh meshes), '
             'pcd (meshes + scene point cloud), interactive (Open3D viewer)'
    )

    args = parser.parse_args()

    completion_output_path = os.path.join(
        args.folder, 'results', 'completion_output'
    )

    if not os.path.exists(completion_output_path):
        raise FileNotFoundError(
            f"completion_output folder not found at {completion_output_path}"
        )

    print(f"Loading from: {completion_output_path}")

    if args.mode == 'all':
        print("\n=== Interactive Scene Visualization ===")
        visualize_scene_with_objects(completion_output_path)
        print("\n=== Meshes + Scene Point Cloud ===")
        visualize_objs_scene_pcd(completion_output_path)
        print("\n=== Colored Meshes (Trimesh) ===")
        visualize_colored_meshes(completion_output_path)
    elif args.mode == 'colored':
        visualize_colored_meshes(completion_output_path)
    elif args.mode == 'pcd':
        visualize_objs_scene_pcd(completion_output_path)
    elif args.mode == 'interactive':
        visualize_scene_with_objects(completion_output_path)


if __name__ == "__main__":
    main()