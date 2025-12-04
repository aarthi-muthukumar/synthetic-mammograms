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
        print(f"File does not exist: {roi_path}")
        continue


    if img is None:
        print(f"Skipping {img_path}: missing image.")
        continue

    if mask is None:
        print(f"Skipping {roi_path}: missing mask")
        continue

    bbox = get_bounding_box_from_mask(mask)
    if bbox is None:
        print(f"No white region found in {roi_path}. Skipping.")
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
