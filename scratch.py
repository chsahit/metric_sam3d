import sys
import os

# Skip sam3d_objects initialization (required for lightweight import)
os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.mesh_from_image_mask import InferenceSequential, load_image, load_mask
from PIL import Image
import numpy as np

image_path = "captures/tab2/rgb.png"
mask_bw_path = "captures/tab2/segmentation_output/sam_outputs/0_object_mask.png"

# Load the original image and mask
img = Image.open(image_path).convert("RGBA")
mask_bw = Image.open(mask_bw_path).convert("L")  # Convert to grayscale

# Use the mask as the alpha channel
# White (255) in mask = opaque, Black (0) = transparent
img.putalpha(mask_bw)

# Save the masked image
masked_image_path = "captures/tab2/masked_image.png"
img.save(masked_image_path)
print(f"Masked image saved to: {masked_image_path}")

# Load model
tag = "hf"
_package_root = os.path.join(os.path.dirname(__file__), "sam-3d-objects")
config_path = os.path.join(_package_root, f"checkpoints/{tag}/pipeline.yaml")
print("Loading model with sequential pipeline (memory-efficient mode)...")
model = InferenceSequential(config_path, compile=False)

# Load image and mask for inference
image = load_image(image_path)
mask = load_mask(masked_image_path)

# Run inference with texture baking enabled
print("Running inference...")
output = model(image, mask, seed=42)

# Save Gaussian splat PLY
ply_output_path = "captures/tab2/completed_rgb.ply"
output["gs"].save_ply(ply_output_path)
print(f"Gaussian splat saved at {ply_output_path}")

# Save mesh OBJ (the mesh is in output["glb"] as a trimesh object)
obj_output_path = "captures/tab2/completed_rgb.obj"
output["glb"].export(obj_output_path)
print(f"Mesh OBJ saved at {obj_output_path}")
