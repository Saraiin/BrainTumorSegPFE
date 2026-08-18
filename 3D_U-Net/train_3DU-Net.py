# =======================================================================
# 3D U-Net Baseline Model Definition, Loss & Optimizer
# =======================================================================

import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

# 1. Set random seed for reproducibility (essential for your Master's thesis report)
set_determinism(seed=42)

# 2. Instantiate the 3D U-Net Baseline architecture
# - in_channels = 4 : Corresponds to our 4 MRI modalities (T1, T1ce, T2, FLAIR)
# - out_channels = 5 : Corresponds to BraTS 2021 labels (0: background, 1: necrosis, 2: edema, 4: enhancing tumor).
#   Since the max index is 4, we configure 5 channels so that automatic One-Hot encoding works without errors.
model = UNet(
    spatial_dims=3,                      # Native 3D volume processing
    in_channels=4,                       # 4 input MRI scans
    out_channels=5,                      # Automatic encoding of BraTS classes
    channels=(16, 32, 64, 128, 256),     # Number of filters at each network level
    strides=(2, 2, 2, 2),                # Downsampling resolution reduction factor
    num_res_units=2,                     # Added residual connections to stabilize gradients
).to(device)

print(f"3D U-Net model successfully initialized and loaded onto GPU: {next(model.parameters()).device}")

# 3. Define the Hybrid Loss Function
# DiceCELoss combines the robustness of the Dice score against class imbalance
# with the voxel-level stability of Cross-Entropy (CE).
loss_function = DiceCELoss(to_onehot=5, softmax=True) # or to_onehot_y=True based on your exact pipeline

# 4. Define the Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

# 5. Final Evaluation Metric: Dice Score (Global mean)
# Exclude the background (include_background=False) to measure accuracy strictly on tumor regions
dice_metric = DiceMetric(include_background=False, reduction="mean")

print("Technical specifications validated: DiceCELoss, Adam Optimizer, and Dice Metric are operational!")


#---------------------------------------------------------------------------------------
# --- 3D U-Net Training, Validation, and Checkpoint Saving Loop ---
# =======================================================================


import os
import time
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

# 1. Directory and post-processing tools configuration
# Path updated to use the new 70/15/15 split directory
checkpoint_dir = '/content/drive/MyDrive/PFE_BraTS_70_15_15'
checkpoint_path = os.path.join(checkpoint_dir, "best_metric_model.pth")

# Transform model outputs into binary masks (One-Hot) for Dice calculation
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# 2. Initialize tracking variables
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print("Starting 3D U-Net Baseline training on L4 GPU...")
print(f"Checkpoints will be saved to: {checkpoint_path}\n")

# 3. Main training loop across epochs
for epoch in range(CONFIG["max_epochs"]):
    print("-" * 50)
    print(f"Epoch {epoch + 1}/{CONFIG['max_epochs']}")
    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()

    # --- TRAINING PHASE ---
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute hybrid Dice + Cross-Entropy loss
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"Batch {step}/{len(train_loader)} - Train Loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    epoch_time = time.time() - start_time
    print(f"[Epoch {epoch + 1} Completed] Mean Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s")

    # --- VALIDATION PHASE (Every 'val_interval' epochs) ---
    if (epoch + 1) % CONFIG["val_interval"] == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                # Sliding window inference to handle large 3D volumes
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=CONFIG["roi_size"],
                    sw_batch_size=4,
                    predictor=model
                )

                # Post-process tensors for metric calculation
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                # Compute Dice score for this patient
                dice_metric(y_pred=val_outputs, y=val_labels)

            # Extract mean Dice score over the validation set
            metric = dice_metric.aggregate().item()
            dice_metric.reset()  # Reset for the next validation session
            metric_values.append(metric)

            print(f"EVALUATION - Validation Mean Dice: {metric:.4f}")

            # --- CHECKPOINT LOGIC ---
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                # Physical weights saving to Google Drive
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved! New best Dice score: {best_metric:.4f}")
            else:
                print(f"No improvement. Best Dice so far: {best_metric:.4f} (Epoch {best_metric_epoch})")

print("\nTraining completed!")
print(f"Best Dice score achieved: {best_metric:.4f} at Epoch {best_metric_epoch}")