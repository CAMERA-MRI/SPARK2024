# 3D Brain Tumor Segmentation with Attention ResUNet

This repository contains a 3D brain tumor segmentation model using a Residual U-Net with attention mechanisms. The model is designed to work with the BraTS dataset and employs a combined loss function of Focal Loss and Dice Coefficient for training.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Loss Functions and Metrics](#loss-functions-and-metrics)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/3d-brain-tumor-segmentation.git
    cd 3d-brain-tumor-segmentation
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

- Place your training images and masks in the respective directories:
    ```
    /path/to/train/images
    /path/to/train/masks
    /path/to/val/images
    /path/to/val/masks
    ```

### Running the Training Script

1. Update the paths in the `train.py` script to point to your dataset and desired output directories.
2. Run the training script:
    ```bash
    python train.py
    ```

## Model Architecture

The model architecture is defined in `model.py` and consists of a Residual U-Net with attention mechanisms.


## Loss Functions and Metrics
The loss functions and metrics are defined in losses_and_metrics.py.

   - Focal Loss: Designed to address class imbalance.
   - Dice Coefficient: Measures overlap between predicted and true segmentation.
   - IoU (Intersection over Union): Measures the accuracy of the segmentation.

## Training
The training process is handled in train.py. It includes data loading, model training, validation, and saving metrics and the trained model.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.

- Fork the repository

- Create a new branch (git checkout -b feature-branch)

- Commit your changes (git commit -am 'Add new feature')

- Push to the branch (git push origin feature-branch)

- Open a Pull Request
