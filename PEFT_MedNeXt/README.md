This repository contains the implementation of the MedNeXt architecture with parameter-efficient fine-tuning (PEFT) using convolutional adapters for brain tumor segmentation. For more details: https://arxiv.org/pdf/2412.14100

# 📌 Overview
This project addresses the segmentation of multi-modal 3D MRI volumes, where the input consists of four MRI sequences: T1-weighted (T1w), T1-weighted contrast-enhanced (T1-c), T2-weighted (T2w), and T2-weighted FLAIR.

The model outputs segmentation masks identifying four classes: Background, Enhancing Tumor (ET), Non-Enhancing Tumor Core (NETC), and Surrounding Non-Enhancing FLAIR Hyperintensity (SNFH).

# 📌 Environment Setup
To set up the environment, clone this repository and install the dependencies from requirements.txt

# 📌 Dataset Preparation
We use two publicly available datasets:
* BraTS 2021 training data: 1251 adult glioma cases
* BraTS Africa training data: 60 adult glioma cases with lower spatial resolution and unique characteristics such as late presentation

## Data Preparation for Sub-Saharan African 2023 Dataset
The BraTS2023_SSA directory looks like this:
```
BraTS2023_SSA
|-- BraTS2023_SSA_Training
|   |--ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2
|   |   |-- BraTS-SSA-00002-000
|   |   |   |-- BraTS-SSA-00002-000-seg.nii.gz
|   |   |   |-- BraTS-SSA-00002-000-t1c.nii.gz
|   |   |   |-- BraTS-SSA-00002-000-t1n.nii.gz
|   |   |   |-- BraTS-SSA-00002-000-t2f.nii.gz
|   |   |   |-- BraTS-SSA-00002-000-t2w.nii.gz
|   |   |-- BraTS-SSA-00007-000
|   |   |   |-- BraTS-SSA-00007-000-seg.nii.gz
|   |   |   |-- BraTS-SSA-00007-000-t1c.nii.gz
|   |   |   |-- BraTS-SSA-00007-000-t1n.nii.gz
|   |   |   |-- BraTS-SSA-00007-000-t2f.nii.gz
|   |   |   |-- BraTS-SSA-00007-000-t2w.nii.gz
|   |   |--...
```

## Stacking the Files
We have 60 folders in the `BraTS2023_SSA_Training/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2` and 15 folders under `BraTS2023_SSA/BraTS2023_SSA_Validation` directory. Each folder contains 5 NIfTI files: one segmentation mask and four image modalities (t1c, t1n, t2f, t2w).

To process these files, use the `stack_ssa.py` script located in `data_prepare_utils/SSA23`. This script requires two parameters: `source_dir` and `flag`.

We need to run `stack_ssa.py` separately for training, validation, and test samples. In our case, we have training and validation sets only, so will run the script twice:

1. For training sets, do:
      ```bash
      python stack_ssa.py --source_dir=full_path_of_ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 --flag=train
      ```
      This creates _TrainVolumes_ and _TrainSegmentations_ folders under a _stacked_ directory at the location where stack_ssa.py is run.
      _TrainVolumes_ contains 60 stacked_nifti image files, and _TrainSegmentations_ contains corresponding 60 segmentation masks.
2. For validation sets, do:
     ```bash
     python stack_ssa.py --source_dir=full_path_of_BraTS2023_SSA_Validation --flag=val
     ```
     This creates _ValVolumes_ and _ValSegmentations_ folders under the same _stacked_ directory.
     _ValVolumes_ contains 15 stacked_nifti image files, and _ValSegmentations_ contains corresponding 15 segmentation masks.

Finally, `stacked` directory looks something like:
```
stacked
|-- TrainVolumes
|   |-- BraTS-SSA-00002-000_stacked.nii.gz
|   |-- BraTS-SSA-00007-000_stacked.nii.gz
|   |-- BraTS-SSA-00008-000_stacked.nii.gz
|   |-- ...
|-- TrainSegmentations
|   |-- BraTS-SSA-00002-000.nii.gz
|   |-- BraTS-SSA-00007-000.nii.gz
|   |-- BraTS-SSA-00008-000.nii.gz
|   |-- ...
|-- ValVolumes
|   |-- BraTS-SSA-00126-000_stacked.nii.gz
|   |-- BraTS-SSA-00129-000_stacked.nii.gz
|   |-- BraTS-SSA-00132-000_stacked.nii.gz
|   |-- ...
|-- ValSegmentations
|   |-- BraTS-SSA-00126-000.nii.gz
|   |-- BraTS-SSA-00129-000.nii.gz
|   |-- BraTS-SSA-00132-000.nii.gz
|   |-- ...
```

## Data Preparation for BraTS 2021 Dataset
Download the BraTS 2021 dataset. The directory looks similar to SSA23 dataset directory, except the number of samples in BraTS2021 is comparatively higher.
```
BraTS2021
|-- BraTS2021_00000
|   |-- BraTS2021_00000_seg.nii.gz
|   |-- BraTS2021_00000_t1.nii.gz
|   |-- BraTS2021_00000_t1ce.nii.gz
|   |-- BraTS2021_00000_t2.nii.gz
|   |-- BraTS2021_00000_flair.nii.gz
|-- BraTS2021_00002
|   |-- BraTS2021_00002_seg.nii.gz
|   |-- BraTS2021_00002_t1.nii.gz
|   |-- BraTS2021_00002_t1ce.nii.gz
|   |-- BraTS2021_00002_t2.nii.gz
|   |-- BraTS2021_00002_flair.nii.gz
...
```

  ## Train Test Split
* Split the dataset into train, validation, and test sets using `train_test_split.py` script provided under `data_prepare_utils/Brats21`.
* `train_test_split.py` takes two parameters: `source_dir` and `train_ratio`
* First, lets split the whole dataset into train and test sets. For this do:
  ```bash
  python train_test_split.py --source_dir=full_path_of_BraTS2021 --train_ratio=0.8 
  ```
  This creates two folders: `train_subset` and `test_subset`
* Now, Rename the ```test_subset to val_subset``` and ```train_subset to subset```
* Second, Split the ```subset``` folder into train and test sets. For this do:
  ```bash
  python train_test_split_ssa.py --source_dir=full_path_of_subset_folder_just_created_above --train_ratio 0.8
  ```
* Delete the ```subset``` folder, if you like.
  Finally, we have 3 folders: train_subset (~60%), val_subset (~20%), test_subset(~20%)

## Remapping labels and Stacking
The label for Enhacing Tumor Region in BraTS 2021 and SSA23 is not same. BraTS 2021 has labelled Enhacing Tumor Region as label 4 in the segmentation mask, while SSA23 has label 3 for it.
So, we need to remap the class 4 to 3, to make it consistent with SSA23 dataset.

Also, We need to stack the four modalities into one.

Both task is done within the script `stack_and_remap_class_brats21.py` provided under `data_prepare_utils/Brats21`. Similar to `stack_ssa.py`, it takes two parameters: `source_dir` and `flag`.

Here, we have train_subset, val_subset, and test_subsets, so we need to run `stack_and_remap_class_brats21.py` three times.
1. For remapping and stacking train_subset, do:
     ```bash
     python stack_and_remap_class_brats21.py --source_dir=full_path_of_train_subset_created_earlier --flag=train
     ```
2. For val_subset, do:
     ```bash
     python stack_and_remap_class_brats21.py --source_dir=full_path_of_val_subset_created_earlier --flag=val
     ```
3. For test_subset, do:
     ```bash
     python stack_and_remap_class_brats21.py --source_dir=full_path_of_test_subset_created_earlier --flag=test
     ```
Finally, we have `stacked` directory that is ready to be used in the project. Looks similar to SSA23 stacked directory.

```
stacked
|-- TrainVolumes
|-- TrainSegmentations
|-- ValVolumes
|-- ValSegmentations
|-- TestVolumes
|-- TestSegmentations
```

# 📌 Training
## Pre-training on BraTS 2021 dataset
To pre-train the model on the BraTS 2021 dataset:
```bash
python src/train.py experiment=brats_mednextv1_small.yaml logger=wandb ++logger.wandb.mode="online" ++trainer.max_epochs=... ++data.batch_size=... ++data.data_dir="...stacked_directory_path_of_brats2021_dataset"
```
Ensure wandb is setup and suitable epochs, batch_size, data_dir is specified.

## Fine-tuning on BraTS Africa dataset
![image](https://github.com/user-attachments/assets/f93cd716-4f03-48e2-9dc6-236584f1b006)
### Full Fine-tuning
Fine-tune all parameters on the SSA dataset with:
```bash
python src/train.py experiment=brats_mednextv1_small.yaml logger=wandb ++logger.wandb.mode="online" ++trainer.max_epochs=... ++data.batch_size=...  ++data.data_dir="...stacked_directory_path_of_ssa_dataset" ++ckpt_path=”…best_checkpoint_path_of_brats21”
```
![image](https://github.com/user-attachments/assets/12663b0b-f6df-4cec-8a38-c5090d3a71e4)

### Fine-tuning with Parallel Adapter Placement
To fine-tune the model using parallel adapter placement, do:
```bash
python src/train.py experiment=brats_mednextv1_small_with_sequential_convnext_adapter.yaml logger=wandb ++logger.wandb.mode="online" ++trainer.max_epochs=... ++data.batch_size=...  ++data.data_dir="...stacked_directory_path_of_ssa_dataset" ++ckpt_path=”…best_checkpoint_path_of_brats21”
```

### Fine-tuning with Sequential Adapter Placement
Fine-tune the model using sequential adapter placement with:
```bash
python src/train.py experiment=brats_mednextv1_small_with_parallel_convnext_adapter.yaml logger=wandb ++logger.wandb.mode="online" ++trainer.max_epochs=... ++data.batch_size=...  ++data.data_dir="...stacked_directory_path_of_ssa_dataset" ++ckpt_path=”…best_checkpoint_path_of_brats21”
```

If you find this work helpful, please consider citing it as:
```bash
@article{adhikari2024parameter,
  title={Parameter-efficient Fine-tuning for improved Convolutional Baseline for Brain Tumor Segmentation in Sub-Saharan Africa Adult Glioma Dataset},
  author={Adhikari, Bijay and Kulung, Pratibha and Bohaju, Jakesh and Poudel, Laxmi Kanta and Raymond, Confidence and Zhang, Dong and Anazodo, Udunna C and Khanal, Bishesh and Shakya, Mahesh},
  journal={arXiv preprint arXiv:2412.14100},
  year={2024}
}
```

This repository is built on the **Hydra-Lightning template**. Refer to this link for more instructions on how to use and run it : https://github.com/ashleve/lightning-hydra-template
