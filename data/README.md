# Dataset - BraTS 2021 (Brain Tumor Segmentation)

This folder contains the scripts and configuration required to download and set up the dataset used for this Master's Thesis project.

## Dataset Link
You can find and access the official dataset challenge page on Kaggle here:
 **[RSNA-ASNR-MICCAI BraTS 2021 Challenge on Kaggle](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)** 
*(Note: You can also use the dedicated BraTS 2021 dataset versions available on Kaggle).*

## Dataset Description
The dataset consists of multi-parametric MRI scans of brain tumors (Glioma), focusing on preoperative scans. 

Each patient directory contains 4 modalities:
* **T1**: T1-weighted, native scans.
* **T1ce**: T1-weighted, contrast-enhanced scans.
* **T2**: T2-weighted scans.
* **FLAIR**: T2 Fluid Attenuated Inversion Recovery scans.
* **Segmentation Mask**: Ground truth labels containing tumor sub-regions:
  * Label 1: Necrotic and non-enhancing tumor core (NCR)
  * Label 2: Peritumoral edema (ED)
  * Label 4: GD-enhancing tumor (ET)
  * *(Whole Tumor (WT) = 1 + 2 + 4, Tumor Core (TC) = 1 + 4, Enhancing Tumor (ET) = 4)*

## How to Download the Data
1. Make sure you have a Kaggle account.
2. Go to your Kaggle account settings to generate and download your API credentials (`kaggle.json`).
3. Run the Kaggle setup script in this folder with your credentials, or use the Kaggle CLI directly:
   ```bash
   kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification