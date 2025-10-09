import os
import cv2
import numpy as np
import pandas as pd

# Path where each subfolder contains overlay files
output_dir = "/home/aarthi/Code/minkiml/Breast Cancer Project/SAM_Med2D/output_masks"
results = []

# Make a new folder to save comparison outputs
comparison_dir = "/home/aarthi/Code/minkiml/Breast Cancer Project/SAM_Med2D/comparisons"
os.makedirs(comparison_dir, exist_ok=True)

def compare_images(img1, img2, label, save_path):
    # Resize if dimensions differ
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Side-by-side comparison
    side_by_side = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(save_path.replace(".png", "_sidebyside.png"), side_by_side)

    # Difference visualization
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Red-highlight difference
    highlight = img1.copy()
    highlight[diff_mask > 0] = [0, 0, 255]  # red for differences
    cv2.imwrite(save_path.replace(".png", "_highlighted.png"), highlight)

    # Stats
    percent_diff = np.sum(diff_mask > 0) / diff_mask.size * 100
    return {
        "Label": label,
        "Difference (%)": round(percent_diff, 2),
        "Non-zero Pixels": int(np.sum(diff_mask > 0))
    }

# Iterate through each case
for case in os.listdir(output_dir):
    case_folder = os.path.join(output_dir, case)
    if not os.path.isdir(case_folder):
        continue
    
    # for visual differences
    orig_path = os.path.join(case_folder, f"{case}_orig_overlay.png")
    gen_path = os.path.join(case_folder, f"{case}_genMRI_overlay.png")

    # for mask comparisons
    # orig_path = os.path.join(case_folder, f"{case}_orig_mask_0.png")
    # gen_path = os.path.join(case_folder, f"{case}_genMRI_mask_0.png")
    

    if not os.path.exists(orig_path) or not os.path.exists(gen_path):
        continue

    # Load overlays
    orig = cv2.imread(orig_path)
    gen = cv2.imread(gen_path)

    # Compare and save
    save_path = os.path.join(comparison_dir, f"{case}_comparison.png")
    stats = compare_images(orig, gen, label=case, save_path=save_path)
    results.append(stats)

# Save stats CSV
stats_df = pd.DataFrame(results)
stats_df.to_csv(os.path.join(comparison_dir, "overlay_comparison_stats.csv"), index=False)

print(f"✅ Comparison complete. Results saved in: {comparison_dir}")
