import torch
import torch.nn as nn
import torch.nn.functional as F

# Blok konvolusi: dua Conv2D + BatchNorm + ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# U-Net sederhana: encoder-decoder dengan skip connection
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder: ekstraksi fitur dengan downsampling
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Decoder: pemulihan resolusi dengan upsampling + skip connection
        self.dec3 = DoubleConv(256 + 128, 128)  # gabungkan dengan enc2
        self.dec2 = DoubleConv(128 + 64, 64)    # gabungkan dengan enc1

        # Lapisan output akhir
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Jalur encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Jalur decoder dengan interpolasi bilinear
        d3 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.final(d2)