import torch
from torch.utils.data import DataLoader
from .datasets import TomatoLeafDataset
from .models import UNet
from .losses import BCEDiceLoss
from .utils import train_one_epoch, evaluate
from pathlib import Path

# Path untuk menyimpan model
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Pilih device: GPU jika tersedia, fallback ke CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Siapkan dataset dan dataloader untuk train dan validasi
dataset_train = TomatoLeafDataset(
    data_root= "data",
    split_file= "data/splits.json",
    split="train"
)

dataset_val = TomatoLeafDataset(
    data_root= "data",
    split_file= "data/splits.json",
    split="val"
)

# Buat DataLoader dengan batch size kecil (sesuai kapasitas memori)
loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)  # shuffle hanya untuk train
loader_val = DataLoader(dataset_val, batch_size=4)  # tidak perlu shuffle saat validasi

# Inisialisasi model U-Net untuk segmentasi biner (1 channel output)
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = BCEDiceLoss()  # kombinasi BCE + Dice, cocok untuk data tidak seimbang
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Jalankan training selama 20 epoch
best_val_loss = float("inf")

for epoch in range(20):
    train_loss = train_one_epoch(
        model, loader_train, optimizer, criterion, device
    )
    val_loss = evaluate(
        model, loader_val, criterion, device
    )

    print(f"[Leaf] Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "leaf_best.pth"
        )
        print("Saved new best leaf model")