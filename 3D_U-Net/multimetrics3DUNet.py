# =======================================================================
# Multi-Metric Evaluation on Test Loader (3D U-Net)
# 
# Evaluated Metrics:
#   - DICE
#   - IoU (Jaccard)
#   - HD95
#   - Vol. Similarity
#   - Sensibilité
#   - Spécificité
#   - Précision
# =======================================================================

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

print("=== STARTING FINAL MULTI-METRIC EVALUATION (3D U-NET) ===")

# 1. Initialize MONAI metric evaluators (ignoring background, reduction="none" for median extraction)
test_dice_metric_unet = DiceMetric(include_background=False, reduction="none")
test_hd95_metric_unet = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95, reduction="none")

# 2. Post-processing tools
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# Lists to store classification metrics per patient/region
sensitivities_unet = []
specificities_unet = []
precisions_unet = []
vol_similarities_unet = []

print(f"Running inference across {len(test_loader)} test patients...")

# 3. Main evaluation loop on the test set
with torch.no_grad():
    for step, test_data in enumerate(test_loader):
        test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)

        # Sliding window inference using the 3D U-Net model
        with torch.amp.autocast('cuda'):
            test_outputs = sliding_window_inference(
                inputs=test_inputs, roi_size=CONFIG["roi_size"], sw_batch_size=2, predictor=model_unet
            )

        preds = decollate_batch(test_outputs)
        labels = decollate_batch(test_labels)

        brats_preds = []
        brats_labels = []

        for p, l in zip(preds, labels):
            p_discrete = post_pred(p)
            l_discrete = post_label(l)

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

            # Stack into final clinical regions tensor [Dummy channel, WT, TC, ET]
            brats_p = torch.stack([torch.zeros_like(et_p), wt_p, tc_p, et_p], dim=0)
            brats_l = torch.stack([torch.zeros_like(et_l), wt_l, tc_l, et_l], dim=0)

            # Transfer to CPU to avoid NVRTC compilation bugs
            brats_preds.append(brats_p.cpu())
            brats_labels.append(brats_l.cpu())

            # --- PIXEL-WISE CLASSIFICATION METRICS CALCULATION ---
            p_np = brats_p.cpu().numpy().astype(np.int8)
            l_np = brats_l.cpu().numpy().astype(np.int8)

            p_m, s_m, pr_m, vs_m = [], [], [], []
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

                p_m.append(sens)
                s_m.append(spec)
                pr_m.append(prec)
                vs_m.append(vs)

            sensitivities_unet.append(p_m)
            specificities_unet.append(s_m)
            precisions_unet.append(pr_m)
            vol_similarities_unet.append(vs_m)

        # Accumulate into MONAI metric evaluators
        test_dice_metric_unet(y_pred=brats_preds, y=brats_labels)
        test_hd95_metric_unet(y_pred=brats_preds, y=brats_labels)

# 4. Extract final NumPy arrays [Patients, 3_Regions]
dice_np = np.nan_to_num(test_dice_metric_unet.aggregate().cpu().numpy(), nan=0.0)
hd95_np = np.nan_to_num(test_hd95_metric_unet.aggregate().cpu().numpy(), nan=0.0)
test_dice_metric_unet.reset()
test_hd95_metric_unet.reset()

# Deduce IoU (Jaccard) from Dice
iou_np = dice_np / (2.0 - dice_np)
sens_np = np.array(sensitivities_unet)
spec_np = np.array(specificities_unet)
prec_np = np.array(precisions_unet)
vs_np = np.array(vol_similarities_unet)

regions = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]

print("\n" + "="*60)
print("     DEFINITIVE MULTI-METRIC REPORT (3D U-NET)     ")
print("="*60)

for idx, region_name in enumerate(regions):
    print(f"\n {region_name} :")
    print(f"  -> DICE :            Mean = {np.mean(dice_np[:, idx]):.4f} | Median = {np.median(dice_np[:, idx]):.4f}")
    print(f"  -> IoU (Jaccard) :   Mean = {np.mean(iou_np[:, idx]):.4f} | Median = {np.median(iou_np[:, idx]):.4f}")
    print(f"  -> HD95 :            Mean = {np.mean(hd95_np[:, idx]):.2f} mm | Median = {np.median(hd95_np[:, idx]):.2f} mm")
    print(f"  -> Vol. Similarity : Mean = {np.mean(vs_np[:, idx]):.4f} | Median = {np.median(vs_np[:, idx]):.4f}")
    print(f"  -> Sensibilité :     Mean = {np.mean(sens_np[:, idx]):.4f} | Median = {np.median(sens_np[:, idx]):.4f}")
    print(f"  -> Spécificité :     Mean = {np.mean(spec_np[:, idx]):.4f} | Median = {np.median(spec_np[:, idx]):.4f}")
    print(f"  -> Précision :       Mean = {np.mean(prec_np[:, idx]):.4f} | Median = {np.median(prec_np[:, idx]):.4f}")

print("\n" + "="*60)
print("GLOBAL EVALUATION SUCCESSFULLY COMPLETED!")
print("="*60)