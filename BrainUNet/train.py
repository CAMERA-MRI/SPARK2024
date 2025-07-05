import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
from model import ResUNetAttention
from metrics_and_loss_functions import FocalLoss, dice_coefficient, iou

# Custom transformation functions and utility functions


def to_tensor_and_normalize(image, mean, std):
    """Convert image to tensor and normalize."""
    image = torch.from_numpy(image).float()
    for t, m, s in zip(image, mean, std):
        t.sub_(m).div_(s)
    return image


def re_encode_mask(mask):
    """Re-encode mask labels from [0, 1, 2, 4] to [0, 1, 2, 3]."""
    mask[mask == 4] = 3
    return mask


def percentile_clip(image, lower_percentile=1, upper_percentile=99):
    """Clip image based on percentiles to handle outliers."""
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    image = np.clip(image, lower, upper)
    return image

# Dataset definition


class BraTSDataset(Dataset):
    """BraTS Dataset."""

    def __init__(self, image_dir, mask_dir, transform=None, lower_percentile=1, upper_percentile=99):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(
            image_dir) if f.startswith('image_')])
        self.masks = sorted(
            [f for f in os.listdir(mask_dir) if f.startswith('mask_')])
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        image = percentile_clip(
            image, self.lower_percentile, self.upper_percentile)
        mask = re_encode_mask(mask)

        if self.transform:
            image = self.transform(image)
        mask = torch.from_numpy(mask)

        image = image.permute(3, 0, 1, 2)
        return image, mask

# Custom transformation class


class CustomTransform:
    """Custom transformation class to normalize the image."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return to_tensor_and_normalize(image, self.mean, self.std)


# Define directories (use pseudo paths for privacy)
train_image_dir = '/path/to/train/images'
train_mask_dir = '/path/to/train/masks'
val_image_dir = '/path/to/val/images'
val_mask_dir = '/path/to/val/masks'

# Define transformation with normalization
mean = [0.5] * 3
std = [0.5] * 3
transform = CustomTransform(mean, std)

# Create datasets and data loaders
train_dataset = BraTSDataset(
    image_dir=train_image_dir, mask_dir=train_mask_dir, transform=transform)
val_dataset = BraTSDataset(image_dir=val_image_dir,
                           mask_dir=val_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Verify dataset lengths
print(f'Length of train dataset: {len(train_dataset)}')
print(f'Length of val dataset: {len(val_dataset)}')

# Initialize model, loss function, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResUNetAttention(in_channels=3, out_channels=4).to(device)
criterion = FocalLoss()
# Adjust learning rate if needed
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training loop
num_epochs = 11  # Set the number of epochs
metrics = {
    "epoch": [],
    "iteration": [],
    "train_loss": [],
    "train_dice": [],
    "train_iou": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": []
}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0

    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        train_dice += dice_coefficient(preds, masks)
        train_iou += iou(preds, masks)

        if (i + 1) % 25 == 0:
            train_loss = running_loss / (i + 1)
            train_dice_avg = train_dice / (i + 1)
            train_iou_avg = train_iou / (i + 1)

            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0
            model.eval()
            val_batch_count = 0

            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    val_images, val_masks = val_images.to(
                        device), val_masks.to(device)
                    val_outputs = model(val_images)
                    val_loss_batch = criterion(val_outputs, val_masks).item()
                    val_loss += val_loss_batch

                    val_preds = val_outputs.argmax(dim=1)
                    val_dice += dice_coefficient(val_preds, val_masks)
                    val_iou += iou(val_preds, val_masks)
                    val_batch_count += 1

            val_loss = val_loss / val_batch_count
            val_dice = val_dice / val_batch_count
            val_iou = val_iou / val_batch_count

            print(f"Epoch {epoch+1}, Iteration {i+1}, Train Loss: {
                  train_loss:.4f}, Train Dice: {train_dice_avg:.4f}, Train IoU: {train_iou_avg:.4f}")
            print(f"Epoch {epoch+1}, Iteration {i+1}, Val Loss: {
                  val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

            metrics["epoch"].append(epoch + 1)
            metrics["iteration"].append(i + 1)
            metrics["train_loss"].append(train_loss)
            metrics["train_dice"].append(train_dice_avg)
            metrics["train_iou"].append(train_iou_avg)
            metrics["val_loss"].append(val_loss)
            metrics["val_dice"].append(val_dice)
            metrics["val_iou"].append(val_iou)

            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(
                '/path/to/save/metrics/metrics_fl_3D.csv', index=False)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    val_batch_count = 0

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            val_loss_batch = criterion(val_outputs, val_masks).item()
            val_loss += val_loss_batch

            val_preds = val_outputs.argmax(dim=1)
            val_dice += dice_coefficient(val_preds, val_masks)
            val_iou += iou(val_preds, val_masks)
            val_batch_count += 1

    val_loss = val_loss / val_batch_count
    val_dice = val_dice / val_batch_count
    val_iou = val_iou / val_batch_count

    print(f"Epoch {epoch+1}, Final Val Loss: {val_loss:.4f}, Final Val Dice: {
          val_dice:.4f}, Final Val IoU: {val_iou:.4f}")

# Save the final model
torch.save(model.state_dict(), '/path/to/save/model/final_model_fl_3D.pth')

# To run the training script:
# python train.py
