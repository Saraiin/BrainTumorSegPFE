# =======================================================================
# Multi-Metric Evaluation on Test Loader (Swin-UNETR)
# [ Evaluated Metrics:
#   - DICE
#   - IoU (Jaccard)
#   - HD95
#   - Vol. Similarity
#   - Sensibilité
#   - Spécificité
#   - Précision]
# =======================================================================

import os
import torch
import numpy as np
import gc
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

print("=== STARTING FINAL MULTI-METRIC EVALUATION ===")

# 1. Load optimal model weights
checkpoint_path = '/content/drive/MyDrive/PFE_BraTS_70_15_15/best_swin_unetr_model.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 2. Original post-processing tools
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# 3. Initialize MONAI metric tools (channel 0 ignored, reduction="none" to extract median)
test_dice_metric = DiceMetric(include_background=False, reduction="none")
test_hd95_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95, reduction="none")

# Lists to store classification metrics per patient/region
sensitivities = []
specificities = []
precisions = []
vol_similarities = []

gc.collect()
torch.cuda.empty_cache()

print("Extracting all metrics on the TEST dataset (Secure CPU execution)...")

# 4. Main inference loop
with torch.no_grad():
    for step, test_data in enumerate(test_loader):
        test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)

        # Sliding window inference optimized for L4 GPU
        with torch.amp.autocast('cuda'):
            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=CONFIG["roi_size"],
                sw_batch_size=2,
                predictor=model
            )

        preds = decollate_batch(test_outputs)
        labels = decollate_batch(test_labels)

        brats_preds = []
        brats_labels = []

        for p, l in zip(preds, labels):
            p_discrete = post_pred(p)  # [5, H, W, D]
            l_discrete = post_label(l)  # [5, H, W, D]

            # --- EXACT COMBINATION OF BRATS REGIONS ---
            # WT Region (Whole Tumor) = C1 + C2 + C4
            wt_p = (p_discrete[1] + p_discrete[2] + p_discrete[4]).clamp(0, 1)
            wt_l = (l_discrete[1] + l_discrete[2] + l_discrete[4]).clamp(0, 1)

            # TC Region (Tumor Core) = C1 + C4
            tc_p = (p_discrete[1] + p_discrete[4]).clamp(0, 1)
            tc_l = (l_discrete[1] + l_discrete[4]).clamp(0, 1)

            # ET Region (Enhancing Tumor) = C4
            et_p = p_discrete[4]
            et_l = l_discrete[4]

            # Stack [Ignored dummy channel, WT, TC, ET]
            brats_p = torch.stack([torch.zeros_like(et_p), wt_p, tc_p, et_p], dim=0)
            brats_l = torch.stack([torch.zeros_like(et_l), wt_l, tc_l, et_l], dim=0)

            # Transfer to CPU to avoid NVRTC compilation bugs
            brats_preds.append(brats_p.cpu())
            brats_labels.append(brats_l.cpu())

            # --- PIXEL-WISE CLASSIFICATION METRICS CALCULATION ---
            p_np = brats_p.cpu().numpy().astype(np.int8)
            l_np = brats_l.cpu().numpy().astype(np.int8)

            p_metrics, s_metrics, pr_metrics, vs_metrics = [], [], [], []

            # Loop over indices 1 (WT), 2 (TC), 3 (ET)
            for r in [1, 2, 3]:
                pred_flat = p_np[r]
                label_flat = l_np[r]

                tp = np.sum((pred_flat == 1) & (label_flat == 1))
                tn = np.sum((pred_flat == 0) & (label_flat == 0))
                fp = np.sum((pred_flat == 1) & (label_flat == 0))
                fn = np.sum((pred_flat == 0) & (label_flat == 1))

                sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                vs = 1.0 - abs(tp + fp - (tp + fn)) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

                p_metrics.append(sens)
                s_metrics.append(spec)
                pr_metrics.append(prec)
                vs_metrics.append(vs)

            sensitivities.append(p_metrics)
            specificities.append(s_metrics)
            precisions.append(pr_metrics)
            vol_similarities.append(vs_metrics)

        # Accumulate into MONAI metric evaluators
        test_dice_metric(y_pred=brats_preds, y=brats_labels)
        test_hd95_metric(y_pred=brats_preds, y=brats_labels)

        if (step + 1) % 20 == 0:
            print(f"Progress: {step + 1}/{len(test_loader)} patients analyzed.")

# 5. Extract final NumPy arrays [Patients, 3_Regions]
dice_np = np.nan_to_num(test_dice_metric.aggregate().cpu().numpy(), nan=0.0)
hd95_np = np.nan_to_num(test_hd95_metric.aggregate().cpu().numpy(), nan=0.0)

test_dice_metric.reset()
test_hd95_metric.reset()

sens_np = np.array(sensitivities)
spec_np = np.array(specificities)
prec_np = np.array(precisions)
vs_np = np.array(vol_similarities)

# Deduce IoU (Jaccard) from Dice
iou_np = dice_np / (2.0 - dice_np)

regions = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]

print("\n" + "="*60)
print("     DEFINITIVE MULTI-METRIC REPORT (Swin-UNETR)     ")
print("="*60)

for idx, region_name in enumerate(regions):
    # Columns of dice_np and hd95_np correspond to WT (0), TC (1), ET (2)
    r_dice = dice_np[:, idx]
    r_iou = iou_np[:, idx]
    r_hd95 = hd95_np[:, idx]
    r_sens = sens_np[:, idx]
    r_spec = spec_np[:, idx]
    r_prec = prec_np[:, idx]
    r_vs = vs_np[:, idx]

    print(f"\n {region_name} :")
    print(f"  - DICE :            Mean = {np.mean(r_dice):.4f} | Median = {np.median(r_dice):.4f}")
    print(f"  - IoU (Jaccard) :   Mean = {np.mean(r_iou):.4f} | Median = {np.median(r_iou):.4f}")
    print(f"  - HD95 :            Mean = {np.mean(r_hd95):.2f} mm | Median = {np.median(r_hd95):.2f} mm")
    print(f"  - Vol. Similarity : Mean = {np.mean(r_vs):.4f} | Median = {np.median(r_vs):.4f}")
    print(f"  - Sensitivity :     Mean = {np.mean(r_sens):.4f} | Median = {np.median(r_sens):.4f}")
    print(f"  - Specificity :     Mean = {np.mean(r_spec):.4f} | Median = {np.median(r_spec):.4f}")
    print(f"  - Precision :       Mean = {np.mean(r_prec):.4f} | Median = {np.median(r_prec):.4f}")

print("\n" + "="*60)
print("GLOBAL EVALUATION SUCCESSFULLY COMPLETED!")
print("="*60)

gc.collect()
torch.cuda.empty_cache()