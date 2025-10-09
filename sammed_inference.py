# import torch
# import os
# import cv2
# import numpy as np
# from segment_anything.build_sam import build_sam_vit_b
# from segment_anything.predictor_sammed import SammedPredictor

# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Define args manually
# class Args:
#     pass

# args = Args()
# args.image_size = 1024
# args.sam_checkpoint = "./pretrain_model/sam-med2d_b.pth"
# args.encoder_adapter = True

# # Build model
# model = build_sam_vit_b(args)
# model.to(device)

# # Predictor
# predictor = SammedPredictor(model)


# # Input and output folders
# input_folder = "/home/aarthi/Code/minkiml/Breast Cancer Project/SAM_Med2D/data_demo/images"  # <-- your test images
# save_folder = "./output_masks"
# os.makedirs(save_folder, exist_ok=True)

# # Predict on each image
# for img_name in os.listdir(input_folder):
#     img_path = os.path.join(input_folder, img_name)
#     img = cv2.imread(img_path)

#     if img is None:
#         print(f"Warning: Could not read image {img_path}")
#         continue

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     predictor.set_image(img_rgb)
#     base_name = os.path.splitext(img_name)[0]  # <--- ADD THIS!


#     # Predict masks
#     masks = predictor.predict()

#     # Check shapes
#     print(f"Shape of masks: {[mask.shape for mask in masks]}")

#     # Initialize valid masks list
#     valid_masks = []

#     # Process each predicted mask
#     for idx, mask in enumerate(masks):
#         mask = np.array(mask)  # ensure it's a NumPy array

#         # Squeeze (1, H, W) -> (H, W)
#         if mask.ndim == 3 and mask.shape[0] == 1:
#             mask = mask.squeeze(0)

#         # Skip if not 2D after squeeze
#         if mask.ndim != 2:
#             print(f"⚠️ Mask {idx} of {img_name} is not 2D after squeeze. Skipping.")
#             continue

#         # Save individual valid mask
#         mask_uint8 = (mask * 255).astype(np.uint8)

#         # Save it
#         individual_folder = os.path.join(save_folder, base_name)
#         os.makedirs(individual_folder, exist_ok=True)

#         mask_path = os.path.join(individual_folder, f"{base_name}_mask_{idx}.png")
#         cv2.imwrite(mask_path, mask_uint8)

#         valid_masks.append(mask_uint8)

#     print(f"Saved {len(valid_masks)} individual masks for {img_name} to {individual_folder}")

#     # 🔥 🔥 🔥 HERE: use **only valid_masks** 🔥 🔥 🔥

#     # Now make the combined mask
#     if len(valid_masks) > 0:
#         combined_mask = np.zeros_like(valid_masks[0], dtype=np.uint8)

#         for idx, mask in enumerate(valid_masks):
#             if mask.shape != combined_mask.shape:
#                 print(f"⚠️ Skipping mask {idx} with shape {mask.shape} for combined mask (expected {combined_mask.shape})")
#                 continue

#             combined_mask = np.maximum(combined_mask, mask)

#         combined_mask_path = os.path.join(save_folder, f"{base_name}_combined.png")
#         cv2.imwrite(combined_mask_path, combined_mask)

#         print(f"Saved combined mask for {img_name} to {combined_mask_path}")
#     else:
#         print(f"⚠️ No valid masks found for {img_name}. No combined mask saved.")


#     # --- Generate pretty colored overlay ---
#     if len(valid_masks) > 0:
#         overlay = img.copy()  # start with the original image (BGR)
#         color_mask = np.zeros_like(overlay, dtype=np.uint8)

#         # Generate a random color for each mask
#         for idx, mask in enumerate(valid_masks):
#             if mask.shape != (img.shape[0], img.shape[1]):
#                 print(f"⚠️ Skipping mask {idx} with shape {mask.shape} for overlay (expected {(img.shape[0], img.shape[1])})")
#                 continue

#             color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # random RGB color
#             colored_region = np.stack([mask, mask, mask], axis=-1)  # make 3 channels
#             colored_region = (colored_region > 0).astype(np.uint8) * color

#             color_mask = cv2.add(color_mask, colored_region)

#         # Blend the color mask onto the original image
#         alpha = 0.5  # transparency factor
#         overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

#         # Save the overlay
#         overlay_path = os.path.join(save_folder, f"{base_name}_overlay.png")
#         cv2.imwrite(overlay_path, overlay)

#         print(f"Saved overlay mask for {img_name} to {overlay_path}")



import os
import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything.build_sam import build_sam_vit_b
from segment_anything.predictor_sammed import SammedPredictor
import re 

def collapse_duplicate_folders(path):
    # Keep collapsing until no duplicate folders remain
    previous = None
    while previous != path:
        previous = path
        path = re.sub(r'/([^/]+)/\1/', r'/\1/', path)
    return path


def fix_double_file_dcm(path):
    parts = path.split('/')
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        return '/'.join(parts[:-1])
    return path


# --- SAMmed Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

class Args:
    pass

args = Args()
args.image_size = 1024
args.sam_checkpoint = "./pretrain_model/sam-med2d_b.pth"
args.encoder_adapter = True

model = build_sam_vit_b(args)
model.to(device)
predictor = SammedPredictor(model)

# --- Load DataFrame ---
df = pd.read_csv("/home/aarthi/Code/minkiml/Breast Cancer Project/SAM_Med2D/data_demo/official_mask_train_set.csv")  # Assumes columns: img_path, roi_path

# --- Output directory ---
output_dir = "./output_masks"
os.makedirs(output_dir, exist_ok=True)

# --- Helper: Get bounding box from mask ---
def get_bounding_box_from_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h

# --- Main Loop ---
for i, row in df.iterrows():
    img_path = os.path.join("/home/aarthi/Code/minkiml/data/cbis-ddsm-postcropfunction", row["cbis_png_path"].lstrip("/") + ".png")
    roi_path = os.path.join("/home/aarthi/Code/minkiml/Breast Cancer Project/SAM_Med2D/data_demo/cbis-masks-postcropfunction", row["cbis_png_maskpath"].lstrip("/") + ".png")
    genmri_path = os.path.join("/home/aarthi/Code/minkiml/data/cbis-generated-mris-postcropfunction", row["cbis_png_path"].lstrip("/") + "_generated.png" )
    print(genmri_path)

    # print(img_path)
    # print(roi_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    mask_out_folder = os.path.join(output_dir, base_name)
    os.makedirs(mask_out_folder, exist_ok=True)

    img = cv2.imread(img_path)
    mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    genmri = cv2.imread(genmri_path)

    if not os.path.exists(roi_path):
        print(f"❌ File does not exist: {roi_path}")
        continue


    if img is None:
        print(f"⚠️ Skipping {img_path}: missing image.")
        continue

    if mask is None:
        print(f"Skipping {roi_path}: missing mask")
        continue

    bbox = get_bounding_box_from_mask(mask)
    if bbox is None:
        print(f"⚠️ No white region found in {roi_path}. Skipping.")
        continue

    x1, y1, x2, y2 = bbox
    # ---- Draw bbox and save it ------
    # Draw bounding box on a copy of the original image
    img_with_box = img.copy()
    genmri_with_box = genmri.copy()
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box, thickness 2
    cv2.rectangle(genmri_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box, thickness 2

    # Save the image with bbox to the same folder as the masks
    cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_bbox.png"), img_with_box)
    cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_genmribbox.png"), genmri_with_box)

    # --- Predict on Original CBIS-DDSM Image ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    input_box = np.array([[x1, y1, x2, y2]])

    orig_masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,
    )

    # Save original image masks
    for idx, mask in enumerate(orig_masks):
        mask_bin = (np.array(mask) > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_orig_mask_{idx}.png"), mask_bin)

    # Combine original masks
    if len(orig_masks) > 0:
        combined_orig = np.zeros_like(orig_masks[0], dtype=np.uint8)
        for mask in orig_masks:
            combined_orig = np.maximum(combined_orig, (mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_orig_combined.png"), combined_orig)

        # Overlay
        color_mask = np.zeros_like(img)
        for mask in orig_masks:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            binary = (mask > 0).astype(np.uint8)
            for c in range(3):
                color_mask[:, :, c][binary == 1] = color[c]
        overlay = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)
        cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_orig_overlay.png"), overlay)

    # --- Predict on Generated MRI ---
    genmri_rgb = cv2.cvtColor(genmri, cv2.COLOR_BGR2RGB)
    predictor.set_image(genmri_rgb)
    input_box = np.array([[x1, y1, x2, y2]])

    gen_masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None, 
        box=input_box,
        multimask_output=True,
    )

    # Save generated MRI masks
    for idx, mask in enumerate(gen_masks):
        mask_bin = (np.array(mask) > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_genMRI_mask_{idx}.png"), mask_bin)

    # Combine generated MRI masks
    if len(gen_masks) > 0:
        combined_gen = np.zeros_like(gen_masks[0], dtype=np.uint8)
        for mask in gen_masks:
            combined_gen = np.maximum(combined_gen, (mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_genMRI_combined.png"), combined_gen)

        # Overlay on generated MRI
        color_mask = np.zeros_like(genmri)
        for mask in gen_masks:
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            binary = (mask > 0).astype(np.uint8)
            for c in range(3):
                color_mask[:, :, c][binary == 1] = color[c]
        overlay = cv2.addWeighted(genmri, 1.0, color_mask, 0.5, 0)
        cv2.imwrite(os.path.join(mask_out_folder, f"{base_name}_genMRI_overlay.png"), overlay)