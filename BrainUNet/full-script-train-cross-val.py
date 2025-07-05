import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class DoubleConv(nn.Module):
    """Double Convolution Block with Batch Normalization, ReLU, and Residual Connection."""

    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        out = self.double_conv(x)
        out = nn.ReLU(inplace=True)(out + residual)
        return out


class AttentionBlock(nn.Module):
    """Attention Block for the U-Net architecture."""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResUNetAttention(nn.Module):
    """Residual U-Net with Attention."""

    def __init__(self, in_channels, out_channels, crop_func=None):
        super(ResUNetAttention, self).__init__()
        self.crop_tensor = crop_func
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(256, 256, 128)
        self.up_conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(128, 128, 64)
        self.up_conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.up_conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.in_conv(x)
        d1p = self.pool(d1)

        d2 = self.down1(d1p)
        d2p = self.pool(d2)

        d3 = self.down2(d2p)
        d3p = self.pool(d3)

        bottleneck = self.bottleneck(d3p)

        up3 = self.up3(bottleneck)
        d3 = self.att3(up3, d3)
        up3 = torch.cat([up3, d3], dim=1)
        up3 = self.up_conv3(up3)

        up2 = self.up2(up3)
        d2 = self.att2(up2, d2)
        up2 = torch.cat([up2, d2], dim=1)
        up2 = self.up_conv2(up2)

        up1 = self.up1(up2)
        d1 = self.att1(up1, d1)
        up1 = torch.cat([up1, d1], dim=1)
        up1 = self.up_conv1(up1)

        return self.out_conv(up1)


# Min-Max Scaling
def min_max_scale(image):
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)
    return image

# Dataset Definition


class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(glob.glob(image_dir + '/*-image.npy'))
        self.masks = sorted(glob.glob(mask_dir + '/*-mask.npy'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image and mask
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        # Apply Min-Max scaling
        image = min_max_scale(image)

        # Convert image and mask to tensors
        image = torch.from_numpy(image).permute(
            3, 0, 1, 2).float()  # (C, H, W, D)
        mask = torch.from_numpy(mask)  # (H, W, D)

        return image, mask

# Dice Coefficient


def dice_coefficient(preds, targets, smooth=1e-6):
    preds = preds.argmax(dim=1)
    preds_one_hot = nn.functional.one_hot(
        preds, num_classes=4).permute(0, 4, 1, 2, 3).float()
    targets_one_hot = nn.functional.one_hot(
        targets, num_classes=4).permute(0, 4, 1, 2, 3).float()

    intersection = (preds_one_hot * targets_one_hot).sum(dim=(2, 3, 4))
    union = preds_one_hot.sum(dim=(2, 3, 4)) + \
        targets_one_hot.sum(dim=(2, 3, 4))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# IoU


def iou(preds, targets, smooth=1e-6):
    preds = preds.argmax(dim=1)
    preds_one_hot = nn.functional.one_hot(
        preds, num_classes=4).permute(0, 4, 1, 2, 3).float()
    targets_one_hot = nn.functional.one_hot(
        targets, num_classes=4).permute(0, 4, 1, 2, 3).float()

    intersection = (preds_one_hot * targets_one_hot).sum(dim=(2, 3, 4))
    union = preds_one_hot.sum(dim=(2, 3, 4)) + \
        targets_one_hot.sum(dim=(2, 3, 4)) - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score.mean().item()

# Weighted Tversky Loss


class WeightedTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, class_weights=None):
        super(WeightedTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        y_true = nn.functional.one_hot(y_true.squeeze(
            1), num_classes=y_pred.size(1)).permute(0, 4, 1, 2, 3).float()
        y_pred = nn.functional.softmax(y_pred, dim=1)

        TP = (y_true * y_pred).sum(dim=(2, 3, 4))
        FP = ((1 - y_true) * y_pred).sum(dim=(2, 3, 4))
        FN = (y_true * (1 - y_pred)).sum(dim=(2, 3, 4))

        Tversky_index = (TP + self.smooth) / \
            (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - Tversky_index

        if self.class_weights is not None:
            loss = loss * \
                torch.tensor(self.class_weights, device=y_pred.device)
        return loss.mean()

# Training and Evaluation Routine


def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, fold, save_dir):
    metrics = []
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    best_iou = 0  # Initialize the best IoU to a very low value
    # Path to save the best model
    best_model_path = os.path.join(save_dir, f'best_model_fold_{fold + 1}.pth')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_dice, train_iou = 0, 0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks)
            train_iou += iou(outputs, masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks)
                val_iou += iou(outputs, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        # Update the best model if the current IoU is better
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Updated best model for Fold {fold + 1} at Epoch {epoch + 1} with IoU: {best_iou:.4f}")

        metrics.append([fold + 1, epoch + 1, train_loss,
                       train_dice, train_iou, val_loss, val_dice, val_iou])
        print(f"Fold {fold + 1} Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

    return metrics


# Cross-Validation


def cross_validate(image_dir, mask_dir, num_folds=5, num_epochs=100, batch_size=2, 
                   save_dir='/cross_val_perfomance_folder'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BraTSDataset(image_dir, mask_dir)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    all_metrics = []
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size)

        model = ResUNetAttention(in_channels=3, out_channels=4).to(device)
        criterion = WeightedTverskyLoss(
            alpha=0.7, beta=0.3, class_weights=[0.5, 1.0, 2.0, 1.5])
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        fold_metrics = train_and_evaluate(
            model, criterion, optimizer, train_loader, val_loader, device, num_epochs, fold, save_dir)
        all_metrics.extend(fold_metrics)

    # Save metrics to CSV
    metrics_path = os.path.join(save_dir, 'cross_validation_metrics_100_epochs.csv')
    df = pd.DataFrame(all_metrics, columns=['Fold', 'Epoch', 'Train Loss', 'Train Dice', 'Train IoU', 'Val Loss', 'Val Dice', 'Val IoU'])
    df.to_csv(metrics_path, index=False)

    print(f"Cross-validation complete. Metrics saved to {metrics_path}.")


# Example Usage
data_path = "/your/data/path"
cross_validate(image_dir=data_path, mask_dir=data_path,num_epochs=100, batch_size=2)
