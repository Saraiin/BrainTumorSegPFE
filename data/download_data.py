# --- Secure Kaggle Configuration ---

# =======================================================================
# KAGGLE API SETUP (RELATED TO DOWNLOADING THE BRATS DATASET)
# This cell configures the Kaggle credentials required to download 
# the dataset directly into the training environment
# =======================================================================

import json
import os

# Users must provide their own Kaggle keys.
username = ""
key = ""

# Create the configuration directory
!mkdir -p ~/.kaggle

# Create the kaggle.json file
kaggle_data = {"username": username, "key": key}
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_data, f)

# Secure the file (strictly required by the Kaggle API)
!chmod 600 ~/.kaggle/kaggle.json

print(" Kaggle configuration successfully completed!")