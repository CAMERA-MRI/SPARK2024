import torch
import torch.nn as nn


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
        return out + residual


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

    def __init__(self, in_channels, out_channels):
        super(ResUNetAttention, self).__init__()
        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(256, 512))

        self.bottleneck = DoubleConv(512, 512)

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
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)

        bottleneck = self.bottleneck(d4)

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

# To use the model, import and create an instance in your training script:
# from model import ResUNetAttention
# model = ResUNetAttention(in_channels=3, out_channels=4).to(device)
