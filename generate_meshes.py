import os
os.environ["LIDRA_SKIP_INIT"] = "true"
from sam3d_objects.mesh_from_image_mask import InferenceSequential, load_image, load_mask
from PIL import Image
import numpy as np
import argparse


def generate_meshes(capture_folder: str, mask: str, mask_type: str) -> None:
    """
    Takes a `capture_folder` which contains...
    """
    # Skip sam3d_objects initialization (required for lightweight import)
    # folder = "captures/tab2"
    image_path = f"{capture_folder}/rgb.png"
    # mask_bw_path = f"{folder}/segmentation_output/sam_outputs/0_object_mask.png"
    mask_bw_path = mask
    output_folder = "outputs"

    # Load the original image and mask
    img = Image.open(image_path).convert("RGBA")
    mask_bw = Image.open(mask_bw_path).convert("L")  # Convert to grayscale

    # Use the mask as the alpha channel
    # White (255) in mask = opaque, Black (0) = transparent
    img.putalpha(mask_bw)

    # Save the masked image
    masked_image_path = f"{output_folder}/masked_image.png"
    img.save(masked_image_path)
    print(f"Masked image saved to: {masked_image_path}")

    # Load model
    tag = "hf"
    _package_root = os.path.join(os.path.dirname(__file__), "sam-3d-objects")
    config_path = os.path.join(_package_root, f"checkpoints/{tag}/pipeline.yaml")
    print("Loading model with sequential pipeline (memory-efficient mode)...")
    model = InferenceSequential(config_path, compile=False, device="cuda:1")

    # Load image and mask for inference
    image = load_image(image_path)
    mask = load_mask(masked_image_path)

    # Run inference with texture baking enabled
    print("Running inference...")
    output = model(image, mask, seed=42)

    # Save Gaussian splat PLY
    ply_output_path = f"{output_folder}/completed_rgb.ply"
    output["gs"].save_ply(ply_output_path)
    print(f"Gaussian splat saved at {ply_output_path}")

    # Save mesh OBJ (the mesh is in output["glb"] as a trimesh object)
    obj_output_path = f"{output_folder}/completed_rgb.obj"
    output["glb"].export(obj_output_path)
    print(f"Mesh OBJ saved at {obj_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from RGB images and masks"
    )
    parser.add_argument(
        "--capture_folder",
        type=str,
        required=True,
        help="Path to the capture folder containing rgb.png"
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the object mask image"
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="alpha",
        help="Type of mask to use (default: alpha)"
    )

    args = parser.parse_args()
    generate_meshes(args.capture_folder, args.mask, args.mask_type)
