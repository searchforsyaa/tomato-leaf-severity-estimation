import torch
from torch.utils.data import DataLoader
from .datasets import TomatoLeafDataset
from .models import UNet
from .losses import BCEDiceLoss
from .utils import evaluate
from pathlib import Path

# Path untuk menyimpan model
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Pilih device: GPU jika tersedia, fallback ke CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Siapkan dataset train dan validasi
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

loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=4)

# Inisialisasi model dan komponen pelatihan
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Pelatihan untuk segmentasi disease (dengan masking leaf)
best_val_loss = float("inf")

for epoch in range(30):
    model.train()
    total_loss = 0

    for img, leaf, disease, _ in loader_train:
        # Pindahkan data ke device
        img = img.to(device)
        leaf = leaf.to(device)
        disease = disease.to(device)

        # Mask gambar hanya pada area daun (non leaf diabaikan)
        img = img * leaf  

        # Langkah optimisasi standar
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, disease)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluasi di akhir epoch
    train_loss = total_loss / len(loader_train)
    val_loss = evaluate(model, loader_val, criterion, device)

    print(f"[Disease] Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "disease_best.pth"
        )
        print("Saved new best disease model")
