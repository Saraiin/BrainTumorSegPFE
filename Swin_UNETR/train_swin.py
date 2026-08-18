# --- Swin-UNETR Hybrid Model Configuration (Dynamic & Secure Version) ---

import os
import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

# 1. Set seed for reproducibility (Master's Thesis requirement)
set_determinism(seed=42)

# 2. Instantiate the Swin-UNETR model
# FIX: Changed feature_size from 48 to 24 to match the Kaggle checkpoint perfectly
model = SwinUNETR(
    spatial_dims=3,                    # Explicitly specify 3D processing
    in_channels=4,                     # 4 MRI modalities (T1, T1ce, T2, FLAIR)
    out_channels=5,                    # 5 output channels for BraTS classes
    feature_size=24,                   # FIXED: Aligned with the weights from epoch 40
    use_checkpoint=True                # Gradient checkpointing: Saves VRAM on the L4 GPU
).to(device)

print(f" Swin-UNETR model successfully initialized on GPU: {next(model.parameters()).device}")

# =======================================================================
# ( RESUMING TRAINING: LOADING WEIGHTS FROM EPOCH 40 ) 
# (Added because the initial training was stopped/interrupted at epoch 40 
# due to Colab limits, so we load the best checkpoint to resume from here)
# =======================================================================

# NOTE: The path below was used during the Master's thesis training 
# environment on Google Colab. 
# TODO: Update this to a local relative path for GitHub usage 
# (e.g., './checkpoints/best_swin_unetr_model.pth') and place the file there.
# checkpoint_path = '/content/drive/MyDrive/PFE_BraTS/best_swin_unetr_model.pth'

# if os.path.exists(checkpoint_path):
    # print(f" Swin-UNETR checkpoint found! Loading weights from: {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location=device)

    # Safety check to extract state_dict if it is wrapped in a dictionary
    # if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
      #  model.load_state_dict(checkpoint["model_state_dict"])
    # else:
        #model.load_state_dict(checkpoint)
    # print(" Epoch 40 weights loaded successfully! Ready to start epoch 41.")
# else:
    #print(f" WARNING: No checkpoint found at: {checkpoint_path}")
    # print(" The model will start from scratch (Epoch 1). Check your Drive path if this is unexpected.")
# =======================================================================

# 3. Loss function and Transformer-specific Optimizer (AdamW)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)

# 4. Evaluation metric for tracking
dice_metric = DiceMetric(include_background=False, reduction="mean")

print(" Ready for the CNN vs Transformer duel!")






# Training Loop for Swin-UNETR Hybrid Model (Optimized for L4 GPU) ---

import time
import torch
import gc
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

max_epochs = CONFIG["max_epochs"]
val_interval = CONFIG["val_interval"]

# start training
start_epoch = 0

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# Post-processing tools to adapt outputs for the Dice metric
post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# Initialize AMP scaler (Accelerated computations in Float16 for L4 GPU)
scaler = torch.cuda.amp.GradScaler()

print(" Starting Swin-UNETR run...")
print(f" Resume scheduled at Epoch {start_epoch + 1}/{max_epochs}")
# TODO: Update the save path below to a local directory (e.g., './checkpoints/...')
print(" Saving best model weights to disk...\n")

for epoch in range(start_epoch, max_epochs):
    print("-" * 50)
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()

    # --- TRAINING PHASE ---
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()

        # Computations wrapped in autocast to optimize VRAM
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

        # Backpropagation and weight updates via AMP Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        if step % 100 == 0:
            print(f"⏱ Step {step}/{len(train_loader)} - Current Loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    elapsed_time = time.time() - start_time
    print(f" Epoch {epoch + 1} completed in {elapsed_time/60:.2f} min - Average Loss: {epoch_loss:.4f}")

    # --- VALIDATION PHASE ---
    if (epoch + 1) % val_interval == 0:
        print("🧹 Preventive VRAM flush before validation...")
        model.eval()

        # Clear training residuals to prevent memory spikes
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                # Sliding window inference (sw_batch_size=2 is ideal for L4 GPU)
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=CONFIG["roi_size"],
                        sw_batch_size=2,
                        predictor=model
                    )

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)

            # Immediate saving of the best model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                
                # NOTE: The path below was used during Colab training. 
                # TODO: Change it to a local path, e.g., './checkpoints/best_swin_unetr_model.pth'
                save_path = '/content/drive/MyDrive/PFE_BraTS/best_swin_unetr_model.pth'
                torch.save(model.state_dict(), save_path)
                print(f"✨ Excellent! Checkpoint updated (Dice: {best_metric:.4f})")

            print(f"Validation Dice Score: {metric:.4f} (Best: {best_metric:.4f} at epoch {best_metric_epoch})")

        # Final cleanup after validation
        gc.collect()
        torch.cuda.empty_cache()

print("\n Swin-UNETR training successfully completed 100%!")