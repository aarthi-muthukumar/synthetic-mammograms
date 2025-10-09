import os
import cv2
import numpy as np
import torch
import glob
import pandas as pd
from scipy.spatial import cKDTree
from monai.transforms import Resize, ScaleIntensity, ToTensor, Compose
from monai.networks.nets import UNet


check_set_dir = "check_set"
os.makedirs(check_set_dir, exist_ok=True)

# Paths
bbox_mask_paths = glob.glob("/home/aarthi/Code/minkiml/Breast Cancer Project/SAM_Med2D/output_masks/*/*_genmribbox.png")
mri_base_path = "/home/aarthi/Code/minkiml/data/cbis-generated-mris-postcropfunction"
output_overlay_dir = "overlay_output"
output_mask_dir = "predicted_crop_masks"
output_dist_eval = "distance_eval"
os.makedirs(output_overlay_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_dist_eval, exist_ok=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(spatial_dims=2, in_channels=1, out_channels=1,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
model.load_state_dict(torch.load("breast_unet_model.pth", map_location=device))
model.eval()

# Helper to get padded shape
def get_padded_shape(shape, k=16):
    h, w = shape
    h_pad = ((h + k - 1) // k) * k
    w_pad = ((w + k - 1) // k) * k
    return (h_pad, w_pad)

# Collect distance info
distance_data = []
summary_stats = []

for mask_path in bbox_mask_paths:
    fname = os.path.basename(mask_path)
    base_name = fname.replace("_genmribbox.png", "")
    mri_path = os.path.join(mri_base_path, f"{base_name}_generated.png")

    if not os.path.exists(mri_path):
        print(f"⚠️ MRI image not found: {mri_path}")
        continue

    mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    bbox_img = cv2.imread(mask_path)

    green_mask = (bbox_img[:, :, 1] > 150) & (bbox_img[:, :, 0] < 100) & (bbox_img[:, :, 2] < 100)
    ys, xs = np.where(green_mask)

    if len(xs) == 0 or len(ys) == 0:
        print(f"❌ No bbox found in: {fname}")
        continue

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    pad = 50
    xmin_p, xmax_p = max(xmin - pad, 0), min(xmax + pad, mri.shape[1])
    ymin_p, ymax_p = max(ymin - pad, 0), min(ymax + pad, mri.shape[0])
    cropped = mri[ymin_p:ymax_p, xmin_p:xmax_p]
    original_shape = cropped.shape  # Save for resizing back later
    target_shape = get_padded_shape(original_shape)

    # Define transform for this image
    preprocess_crop = Compose([
        Resize(target_shape),
        ScaleIntensity(),
        ToTensor()
    ])

    # Apply transform
    arr = np.expand_dims(cropped, axis=0).astype(np.float32)
    input_tensor = preprocess_crop(arr).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
        pred = (output > 0.5).float().cpu().numpy()[0, 0]

    # Resize prediction back to original crop size
    pred_resized = cv2.resize(pred, (original_shape[1], original_shape[0]))
    print(f"[DEBUG] Pred shape: {pred.shape}, resized back to: {pred_resized.shape}, crop shape: {original_shape}")

    
    # Fill into full image mask
    full_mask = np.zeros_like(mri, dtype=np.uint8)
    full_mask[ymin_p:ymax_p, xmin_p:xmax_p] = (pred_resized * 255).astype(np.uint8)

    # Distance computation
    pred_coords = np.column_stack(np.where(full_mask > 127))
    gt_coords = np.array([[ymin, x] for x in range(xmin, xmax + 1)] +
                         [[ymax, x] for x in range(xmin, xmax + 1)] +
                         [[y, xmin] for y in range(ymin, ymax + 1)] +
                         [[y, xmax] for y in range(ymin, ymax + 1)])

    for py, px in pred_coords:
        if ymin <= py <= ymax and xmin <= px <= xmax:
            dist = 0.0
        else:
            dist = np.min(np.sqrt((gt_coords[:, 0] - py) ** 2 + (gt_coords[:, 1] - px) ** 2))
        distance_data.append({"filename": base_name, "px": px, "py": py, "distance_to_bbox": dist})

    # Summary stats for this image
    image_distances = [d["distance_to_bbox"] for d in distance_data if d["filename"] == base_name]
    if image_distances:
        summary_stats.append({
            "filename": base_name,
            "mean_distance": np.mean(image_distances),
            "min_distance": np.min(image_distances),
            "max_distance": np.max(image_distances),
            "num_predicted_pixels": len(image_distances)
        })

    # Save visual overlays
    overlay = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    overlay[full_mask > 127] = [255, 0, 0]
    safe_base = base_name.replace("/", "_")
    cv2.imwrite(os.path.join(output_mask_dir, f"{safe_base}_mask.png"), full_mask)
    cv2.imwrite(os.path.join(output_overlay_dir, f"{safe_base}_overlay.png"), overlay)

    # padded area analysed with MONAI
    # Diagnostic check image (showing padded crop and MONAI mask)
    debug_crop_overlay = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # Resize predicted mask to match original crop shape
    pred_vis = (pred_resized * 255).astype(np.uint8)
    mask_colored = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
    mask_alpha = 0.5

    # Blend predicted mask onto grayscale crop
    blended = cv2.addWeighted(debug_crop_overlay, 1 - mask_alpha, mask_colored, mask_alpha, 0)

    # Save overlay of the area MONAI ran on
    cv2.imwrite(os.path.join(check_set_dir, f"{safe_base}_MONAI_crop_overlay.png"), blended)


# Save CSVs
df = pd.DataFrame(distance_data)
df.to_csv("pixel_distances_to_bbox.csv", index=False)
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv("distance_summary_stats.csv", index=False)

print('All processing and distance evaluation complete.")
