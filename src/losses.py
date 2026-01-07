import torch
import torch.nn as nn

# Dice Loss dengan smoothing untuk menghindari pembagian dengan nol
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Terapkan sigmoid karena prediksi berupa logit
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        # Hitung Dice coefficient lalu ubah ke loss (1 - Dice)
        return 1 - (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

# Kombinasi BCE dan Dice Loss, cocok untuk segmentasi dengan class imbalance
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Sudah termasuk sigmoid di dalamnya
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Jumlahkan kedua loss
        return self.bce(pred, target) + self.dice(pred, target)