# --- 3D U-NET FINAL EVALUATION CELL: OFFICIAL BRATS METRICS (WT, TC, ET) ---

import torch
import numpy as np
import gc
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

# 1. Strictly identical post-processing for a fair comparison
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# 2. Initialize metric calculators (Channel 0 ignored)
dice_brats = DiceMetric(include_background=False, reduction="mean_batch")
hd95_brats = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95, reduction="mean_batch")

# SAFETY: Set the 3D U-Net model to evaluation mode
model.eval()
print(" Extracting official metrics (WT, TC, ET) for the 3D U-Net Baseline...")
print(" Safety CPU computation activated for HD95 (Maximum stability).\n")

with torch.no_grad():
    for val_data in val_loader:
        val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

        # Sliding window inference (Exactly like the Swin-UNETR)
        with torch.amp.autocast('cuda'):
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=CONFIG["roi_size"],
                sw_batch_size=2,
                predictor=model
            )

        # Decollate the batch
        preds = decollate_batch(val_outputs)
        labels = decollate_batch(val_labels)

        brats_preds = []
        brats_labels = []

        for p, l in zip(preds, labels):
            p_discrete = post_pred(p)  # Tensor shape: [5, H, W, D]
            l_discrete = post_label(l)  # Tensor shape: [5, H, W, D]

            # --- LOGICAL MERGING OF BRATS CLINICAL CHANNELS ---
            # Reminder: 1=Necrosis, 2=Edema, 4=Enhancing Tumor

            # WT Region (Whole Tumor) = Necrosis + Edema + Enhancing
            wt_p = (p_discrete[1] + p_discrete[2] + p_discrete[4]).clamp(0, 1)
            wt_l = (l_discrete[1] + l_discrete[2] + l_discrete[4]).clamp(0, 1)

            # TC Region (Tumor Core) = Necrosis + Enhancing
            tc_p = (p_discrete[1] + p_discrete[4]).clamp(0, 1)
            tc_l = (l_discrete[1] + l_discrete[4]).clamp(0, 1)

            # ET Region (Enhancing Tumor) = Enhancing only
            et_p = p_discrete[4]
            et_l = l_discrete[4]

            # Stacking into the final tensor [Dummy channel, WT, TC, ET]
            brats_p = torch.stack([torch.zeros_like(et_p), wt_p, tc_p, et_p], dim=0)
            brats_l = torch.stack([torch.zeros_like(et_l), wt_l, tc_l, et_l], dim=0)

            # Immediate transfer to CPU to bypass CUDA driver bugs
            brats_preds.append(brats_p.cpu())
            brats_labels.append(brats_l.cpu())

        # Local accumulation of results
        dice_brats(y_pred=brats_preds, y=brats_labels)
        hd95_brats(y_pred=brats_preds, y=brats_labels)

# 3. Calculation of global averages on the validation set
scores_dice = dice_brats.aggregate()
scores_hd95 = hd95_brats.aggregate()

dice_brats.reset()
hd95_brats.reset()

# Clean memory release
gc.collect()
torch.cuda.empty_cache()

# --- DISPLAY BLOCK FOR YOUR NUMBERS ---
print("="*60)
print(" FINAL 3D U-NET BASELINE RESULTS :")
print("="*60)
print(f"  [WT] Whole Tumor     - Dice: {scores_dice[0].item():.4f} | HD95: {scores_hd95[0].item():.3f} mm")
print(f"  [TC] Tumor Core      - Dice: {scores_dice[1].item():.4f} | HD95: {scores_hd95[1].item():.3f} mm")
print(f"  [ET] Enhancing Tumor - Dice: {scores_dice[2].item():.4f} | HD95: {scores_hd95[2].item():.3f} mm")
print("="*60)