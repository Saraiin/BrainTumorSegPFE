# ---  3D U-Net Baseline Model Definition, Loss & Optimizer ---

import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

# 1. Set random seed for reproducibility (essential for the Master's Thesis report)
set_determinism(seed=42)

# 2. Instantiate the 3D U-Net Baseline architecture
# - in_channels = 4: corresponds to our 4 MRI modalities (T1, T1ce, T2, FLAIR)
# - out_channels = 5: corresponds to BraTS 2021 labels (0:background, 1:necrosis, 2:edema, 4:enhancing tumor).
#   Since the max index is 4, we configure 5 channels so the automatic One-Hot encoding works without errors.
model = UNet(
    spatial_dims=3,                    # Native processing of 3D volumes
    in_channels=4,                     # 4 input MRI images
    out_channels=5,                    # Automatic encoding of BraTS classes
    channels=(16, 32, 64, 128, 256),   # Number of filters at each network level
    strides=(2, 2, 2, 2),              # Resolution reduction factor (Downsampling)
    num_res_units=2,                   # Adds residual connections to stabilize gradients
).to(device)

print(f" 3D U-Net model successfully initialized and moved to GPU: {next(model.parameters()).device}")

# 3. Define the Hybrid Loss Function
# DiceCELoss combines the robustness of the Dice score against class imbalance 
# and the voxel-level stability of Cross-Entropy (CE).
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

# 4. Define the Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

# 5. Final evaluation metric: Dice Score (Global average)
# Background is excluded (include_background=False) to measure accuracy only on the tumor regions
dice_metric = DiceMetric(include_background=False, reduction="mean")

print(" Technical specifications validated: DiceCELoss, Adam Optimizer, and Dice Metric are operational!")



#---------------------------------------------------------------------------------------
# --- Training, Validation, and Checkpoint Saving Loop ---

import os
import time
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

# 1. Directory configuration and post-processing tools
# NOTE: The path below was used during Colab training. 
# TODO: Update this to a local relative path, e.g., './checkpoints'
checkpoint_dir = '/content/drive/MyDrive/PFE_BraTS'
checkpoint_path = os.path.join(checkpoint_dir, "best_metric_model.pth")

# Transform model outputs into binary masks (One-Hot) for Dice calculation
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# 2. Initialize tracking variables
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print(" Starting 3D U-Net Baseline training on L4 GPU...")
print(f" Checkpoints will be saved at: {checkpoint_path}\n")

# 3. Main Epoch loop
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

        # Calculate hybrid Dice + Cross-Entropy loss
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f" Batch {step}/{len(train_loader)} - Train Loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    epoch_time = time.time() - start_time
    print(f" [Epoch {epoch + 1} Completed] Average Train Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s")

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

                # Calculate Dice score for this patient
                dice_metric(y_pred=val_outputs, y=val_labels)

            # Extract mean Dice score over the validation set
            metric = dice_metric.aggregate().item()
            dice_metric.reset() # Reset for the next validation session
            metric_values.append(metric)

            print(f" EVALUATION - Validation Mean Dice: {metric:.4f}")

            # --- CHECKPOINT LOGIC ---
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                # Save physical weights to disk/Drive
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✨ Checkpoint saved! New best Dice score: {best_metric:.4f}")
            else:
                print(f" No improvement. Best Dice so far: {best_metric:.4f} (Epoch {best_metric_epoch})")

print("\n Training completed!")
print(f" Best Dice score achieved: {best_metric:.4f} at Epoch {best_metric_epoch}")