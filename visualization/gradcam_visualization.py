# =======================================================================
#  Universal Grad-CAM with Patch Matching (Anti-Mismatch)
# Description: Generates 3D saliency maps using MONAI's GradCAM to interpret 
# model predictions on BraTS MRI scans
# =======================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Official MONAI imports
from monai.visualize import GradCAM
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

os.environ["CCCL_IGNORE_DEPRECATED_CPP_DIALECT"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drive_dir = '/content/drive/MyDrive/PFE_BraTS_70_15_15/'

# 💡 Modify if your model instance has a different name in your notebook
UNET_MODEL_INSTANCE = model

# --- 1. AUTOMATIC CHANNEL FORMAT ANALYSIS ---
print("Analyzing output dimensions...")
for test_batch in val_loader:
    test_inputs = test_batch["image"].to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            test_outputs = sliding_window_inference(test_inputs, (128, 128, 128), sw_batch_size=2, predictor=UNET_MODEL_INSTANCE)
    num_channels = test_outputs.shape[1]
    break
print(f"Validated architecture: {num_channels} channels detected.")

# Define region index based on channels
region_id_cam = 2 if num_channels == 3 else 4
region_name_cam = "Active Tumor (ET)" if num_channels == 3 else "Active Tumor (Label 4)"

# --- 2. DYNAMIC TARGET LAYER SEARCH ---
unet_target_layer = None
for name, module in UNET_MODEL_INSTANCE.named_modules():
    if isinstance(module, torch.nn.Conv3d):
        unet_target_layer = name

print(f"Target layer automatically detected: '{unet_target_layer}'")

# --- 3. LOAD MODEL WEIGHTS ---
checkpoint_path = os.path.join(drive_dir, 'best_unet3d_model.pth')
if os.path.exists(checkpoint_path):
    UNET_MODEL_INSTANCE.load_state_dict(torch.load(checkpoint_path, map_location=device))
    UNET_MODEL_INSTANCE.to(device)
    UNET_MODEL_INSTANCE.eval()
    print(f"Success: Weights restored from checkpoint!")
else:
    print(f"Warning: Checkpoint file not found at {checkpoint_path}.")

# --- 4. PREPARE PERFECT-SIZED PATCH (128x128x128) ---
print("Extracting a 128x128x128 volume for Grad-CAM...")
for batch_data in val_loader:
    raw_inputs = batch_data["image"][:1]
    raw_labels = batch_data["label"][:1]
    break

# Central surgical crop to ensure dimension divisibility within the U-Net
B, C, H, W, D = raw_inputs.shape
h_start = (H - 128) // 2
w_start = (W - 128) // 2
d_start = (D - 128) // 2

test_inputs = raw_inputs[:, :, h_start:h_start+128, w_start:w_start+128, d_start:d_start+128].to(device)
test_labels = raw_labels[:, :, h_start:h_start+128, w_start:w_start+128, d_start:d_start+128].to(device)

# --- 5. COMPUTE GRAD-CAM HEATMAP ON THE PATCH ---
print(f"Computing Grad-CAM Heatmap (Zero geometric crash)...")
cam_instance = GradCAM(nn_module=UNET_MODEL_INSTANCE, target_layers=unet_target_layer)

# Tensor size is precisely 128x128x128, direct inference passes without errors
cam_heatmap_3d = cam_instance(x=test_inputs, class_idx=region_id_cam)
cam_heatmap_raw = cam_heatmap_3d[0, 0].detach().cpu().numpy()

# Inference on the same patch
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        test_outputs = UNET_MODEL_INSTANCE(test_inputs)

# Post-processing masks
if num_channels == 3:
    preds_final = (test_outputs[0].sigmoid() > 0.5)[region_id_cam].cpu().numpy()
    labels_final = test_labels[0, region_id_cam].cpu().numpy()
else:
    preds_argmax = torch.argmax(test_outputs, dim=1)
    preds_final = (preds_argmax[0] == region_id_cam).cpu().numpy()
    labels_final = (test_labels[0, 0] == region_id_cam).cpu().numpy()

image_t1ce = test_inputs[0, 0].cpu().numpy()

# --- 6. VISUALIZATION ---
slice_idx = 64  # Exact center of our new 128 patch
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.title("1. Native MRI (T1ce)", fontsize=12, fontweight='bold')
plt.imshow(image_t1ce[:, :, slice_idx], cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("2. Ground Truth (Expert)", fontsize=12, fontweight='bold')
plt.imshow(labels_final[:, :, slice_idx], cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("3. 3D U-Net Prediction", fontsize=12, fontweight='bold')
plt.imshow(preds_final[:, :, slice_idx], cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("4. Grad-CAM Saliency Map", fontsize=12, fontweight='bold', color='crimson')
plt.imshow(image_t1ce[:, :, slice_idx], cmap='gray')
plt.imshow(cam_heatmap_raw[:, :, slice_idx], cmap='jet', alpha=0.45)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

plt.tight_layout()
plt.show()