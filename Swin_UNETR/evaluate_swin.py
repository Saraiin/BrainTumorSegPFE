# =======================================================================
# Final Evaluation on Test Loader (BraTS Metrics: WT, TC, ET)
# =======================================================================

import torch
import numpy as np
import gc
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

# 1. Post-processing tools to binarize model predictions
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# 2. Initialize metric calculators (include_background=False to ignore channel 0)
dice_brats = DiceMetric(include_background=False, reduction="mean_batch")
hd95_brats = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95, reduction="mean_batch")

model.eval()
print("Extracting clinical metrics (WT, TC, ET) for Swin-UNETR...")
print("Official evaluation on the TEST set (Unseen data)...")
print("CPU fallback detection mode activated to bypass CUDA compilation bugs.\n")

with torch.no_grad():
    for test_data in test_loader: # Modified to use the test dataset loader
        test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)

        # Sliding window inference on the L4 GPU
        with torch.amp.autocast('cuda'):
            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=CONFIG["roi_size"],
                sw_batch_size=2,
                predictor=model
            )

        # Decollate the batch
        preds = decollate_batch(test_outputs)
        labels = decollate_batch(test_labels)

        brats_preds = []
        brats_labels = []

        for p, l in zip(preds, labels):
            p_discrete = post_pred(p)  # Shape: [5, H, W, D]
            l_discrete = post_label(l)  # Shape: [5, H, W, D]

            # --- COMBINING CLASSES FOR BRATS CLINICAL REGIONS ---
            # Channel 1 = Necrosis (C1), Channel 2 = Edema (C2), Channel 4 = Enhancing Tumor (C4)

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

            # CRITICAL SAFETY: Transfer tensors to CPU (.cpu())
            # This avoids using CuPy and eliminates the NVRTC_ERROR_COMPILATION bug
            brats_preds.append(brats_p.cpu())
            brats_labels.append(brats_l.cpu())

        # Accumulate results on the CPU
        dice_brats(y_pred=brats_preds, y=brats_labels)
        hd95_brats(y_pred=brats_preds, y=brats_labels)

# 3. Retrieve global averages
scores_dice = dice_brats.aggregate()
scores_hd95 = hd95_brats.aggregate()

dice_brats.reset()
hd95_brats.reset()

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

# --- DISPLAY RESULTS FOR YOUR THESIS REPORT ---
print("="*60)
print(" FINAL SWIN-UNETR RESULTS ON TEST DATA :")
print("="*60)
print(f"   [WT] Whole Tumor     - Dice: {scores_dice[0].item():.4f} | HD95: {scores_hd95[0].item():.3f} mm")
print(f"   [TC] Tumor Core      - Dice: {scores_dice[1].item():.4f} | HD95: {scores_hd95[1].item():.3f} mm")
print(f"   [ET] Enhancing Tumor - Dice: {scores_dice[2].item():.4f} | HD95: {scores_hd95[2].item():.3f} mm")
print("="*60)
print("GENERAL EVALUATION SUCCESSFULLY COMPLETED!")