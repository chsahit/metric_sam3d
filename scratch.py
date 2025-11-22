import sys
import os

# Skip sam3d_objects initialization (required for lightweight import)
os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.mesh_from_image_mask import mesh_from_image_mask
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

# Now use this for the mesh generation
mesh_from_image_mask(image_path, masked_image_path)
