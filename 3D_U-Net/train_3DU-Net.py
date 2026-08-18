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