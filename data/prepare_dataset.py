# =======================================================================
# SCRIPT: prepare_dataset.py
# Description: Scans the raw BraTS 2021 dataset, filters complete patient 
# scans, splits them into Train (70%), Validation (15%), and Test (15%) sets, 
# and saves the mapping to a JSON file.
# =======================================================================

import os
import glob
import json
from google.colab import drive
from sklearn.model_selection import train_test_split

# 1. Mount Google Drive
print("Connecting to Google Drive...")
drive.mount('/content/drive')

# 2. Define paths (Updated for the 70/15/15 split structure)
# TODO FOR GITHUB: Update these paths for your local environment if not using Colab
data_dir = '/content/brats_data'
out_dir = '/content/drive/MyDrive/PFE_BraTS_70_15_15'
json_path = os.path.join(out_dir, 'dataset_brats21_70_15_15.json')

# Create target directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# 3. Automatic scan of BraTS 2021 patient folders
patient_folders = sorted([f for f in glob.glob(os.path.join(data_dir, "**", "BraTS2021_*"), recursive=True) if os.path.isdir(f)])

if len(patient_folders) == 0:
    patient_folders = sorted(list(set([os.path.dirname(p) for p in glob.glob(os.path.join(data_dir, "**", "*_flair.nii*"), recursive=True)])))

print(f"Patient folders detected: {len(patient_folders)}")

# 4. Construct the file list (MRI Modalities + Label)
dataset_dicts = []
for folder in patient_folders:
    p_id = os.path.basename(folder)

    t1 = glob.glob(os.path.join(folder, f"*{p_id}*_t1.nii*"))
    t1ce = glob.glob(os.path.join(folder, f"*{p_id}*_t1ce.nii*"))
    t2 = glob.glob(os.path.join(folder, f"*{p_id}*_t2.nii*"))
    flair = glob.glob(os.path.join(folder, f"*{p_id}*_flair.nii*"))
    seg = glob.glob(os.path.join(folder, f"*{p_id}*_seg.nii*"))

    if t1 and t1ce and t2 and flair and seg:
        dataset_dicts.append({
            "image": [t1[0], t1ce[0], t2[0], flair[0]],
            "label": seg[0]
        })

print(f"Valid and complete patients retained: {len(dataset_dicts)}")

# 5. Split into Train (70%) / Validation (15%) / Test (15%) and save to Drive
if len(dataset_dicts) > 0:
    # Step 1: Extract 70% for training, keeping 30% for validation + test
    train_files, rest_files = train_test_split(dataset_dicts, test_size=0.30, random_state=42)

    # Step 2: Split the remaining 30% equally into validation and test (50% of 30% = 15% each)
    val_files, test_files = train_test_split(rest_files, test_size=0.50, random_state=42)

    # Create the complete dictionary with the 3 splits
    dataset_split = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    with open(json_path, 'w') as f:
        json.dump(dataset_split, f, indent=4)

    print(f"Reference JSON file successfully created: {json_path}")
    print(f"Split breakdown: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")
else:
    print("Error: No valid patients could be indexed. Check your folder structure.")