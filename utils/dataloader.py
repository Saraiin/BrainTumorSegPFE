# =======================================================================
# CELL 5: Ultra-Lightweight Standard Pipeline (ZERO CACHE, BATCH CONFIG 2 - L4 GPU)
# =======================================================================

import os
import json
import torch
from google.colab import drive
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, CropForegroundd, Orientationd,
    Spacingd, NormalizeIntensityd, SpatialPadd, RandCropByPosNegLabeld, RandFlipd, EnsureTyped
)
from monai.data import Dataset, DataLoader

# 1. Security purge of the local Colab disk
print("Security purge... Cleaning up all legacy caches...")
!rm -rf /content/train_cache /content/val_cache /content/train_cache_opt /content/val_cache_opt
print("All caches successfully removed from disk.")

# 2. Standard Google Drive connection and GPU detection
drive.mount('/content/drive')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Machine ready. GPU: {torch.cuda.get_device_name(0)}")

CONFIG = {
    "roi_size": (128, 128, 128),
    "batch_size": 2,                    # Reset to 2: Perfectly handled by the L4 GPU's 24 GB VRAM
    "lr": 2e-4,
    "max_epochs": 100,
    "val_interval": 2,
}

# The path now points to the new 70/15/15 split directory and JSON file
json_path = '/content/drive/MyDrive/PFE_BraTS_70_15_15/dataset_brats21_70_15_15.json'

# 3. Size-optimized pipelines (Background cropping + Padding security)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image", "label"], spatial_size=CONFIG["roi_size"], mode=["constant", "constant"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=CONFIG["roi_size"], num_samples=1, pos=1, neg=1
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image", "label"], spatial_size=CONFIG["roi_size"], mode=["constant", "constant"]),
    EnsureTyped(keys=["image", "label"]),
])

# 4. Load data without cache directly from Google Drive
with open(json_path, 'r') as f:
    dataset_split = json.load(f)

print("\nInitializing Datasets in streaming mode...")
train_ds = Dataset(data=dataset_split["train"], transform=train_transforms)
val_ds = Dataset(data=dataset_split["val"], transform=val_transforms)
test_ds = Dataset(data=dataset_split["test"], transform=val_transforms)

# 5. Secure DataLoaders against RAM leaks (num_workers=0)
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation patients: {len(val_loader)}")
print(f"Number of test patients: {len(test_loader)}")
print("\nData loader configuration successfully completed.")