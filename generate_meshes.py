import os
os.environ["LIDRA_SKIP_INIT"] = "true"
from sam3d_objects.mesh_from_image_mask import InferenceSequential, load_image, load_mask
from PIL import Image
import numpy as np
import argparse
import glob


def generate_meshes(capture_folder: str, output_folder: str, mask_type: str, device: str = "cuda:0") -> None:
    """
    Takes a `capture_folder/` which contains an rgb.png, a depth.png, and an intrinsics.npy
    as well as a masks/ subfolder with image masks (png files)
    Processes multiple masks and generates meshes for each
    """
    # Normalize paths to handle trailing slashes
    capture_folder = os.path.normpath(capture_folder)
    output_folder = os.path.normpath(output_folder)

    image_path = os.path.join(capture_folder, "rgb.png")
    mask_expr = os.path.join(capture_folder, "masks", "*.png")
    masks = list(glob.glob(mask_expr))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load model once (reuse for all masks)
    tag = "hf"
    _package_root = os.path.join(os.path.dirname(__file__), "sam-3d-objects")
    config_path = os.path.join(_package_root, f"checkpoints/{tag}/pipeline.yaml")
    print(f"Loading model with sequential pipeline (memory-efficient mode) on {device}...")
    model = InferenceSequential(config_path, compile=False, device=device)

    # Process each mask with numeric IDs
    for idx, mask_bw_path in enumerate(masks):
        print(f"\n{'='*60}")
        print(f"Processing mask {idx + 1}/{len(masks)}: {mask_bw_path}")
        print(f"{'='*60}")

        # Use numeric ID for output files
        numeric_id = str(idx)

        # Load the original image and mask
        img = Image.open(image_path).convert("RGBA")
        mask_bw = Image.open(mask_bw_path).convert("L")  # Convert to grayscale

        # Use the mask as the alpha channel
        # White (255) in mask = opaque, Black (0) = transparent
        img.putalpha(mask_bw)

        # Save the masked image
        masked_image_path = os.path.join(output_folder, f"masked_image_{numeric_id}.png")
        img.save(masked_image_path)
        print(f"Masked image saved to: {masked_image_path}")

        # Load image and mask for inference
        image = load_image(image_path)
        mask = load_mask(masked_image_path)

        # Run inference
        print("Running inference...")
        output = model(image, mask, seed=42)

        print("saving ply")
        # Save Gaussian splat PLY with numeric ID
        ply_output_path = os.path.join(output_folder, f"{numeric_id}.ply")
        output["gs"].save_ply(ply_output_path)
        print(f"Gaussian splat saved at {ply_output_path}")
        print("saving obj")
        # Save mesh OBJ with numeric ID
        obj_output_path = os.path.join(output_folder, f"{numeric_id}.obj")
        output["glb"].export(obj_output_path)
        print(f"Mesh OBJ saved at {obj_output_path}")

    print(f"\n{'='*60}")
    print(f"Completed processing {len(masks)} masks")
    print(f"{'='*60}")


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
        "--output_folder",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="alpha",
        help="Type of mask to use (default: alpha)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device to use (e.g., '0' for cuda:0, '1' for cuda:1)"
    )

    args = parser.parse_args()

    # Format device string
    device = f"cuda:{args.device}" if not args.device.startswith("cuda:") else args.device

    generate_meshes(args.capture_folder, args.output_folder, args.mask_type, device)
