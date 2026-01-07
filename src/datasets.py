import json
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

# Muat gambar dalam format RGB
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Muat mask biner; jika tidak ada, kembalikan mask nol
def load_mask(path, shape):
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    return mask


# Dataset untuk segmentasi (healthy vs early blight)
class TomatoLeafDataset(Dataset):
    def __init__(
        self,
        data_root,
        split_file,
        split="train",
        image_size=256,
        transform=None
    ):
        """
        data_root : folder utama data
        split_file: file JSON berisi daftar file per split
        split     : pilihan: train / val / test
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.transform = transform

        with open(split_file, "r") as f:
            splits = json.load(f)

        self.samples = []

        # Tambahkan sampel Early Blight
        for fname in splits["early_blight"][split]:
            self.samples.append({
                "class": "early_blight",
                "image": self.data_root / "raw" / "Tomato___Early_blight" / fname,
                "leaf": self.data_root / "masks" / "early_blight" / "leaf" / (Path(fname).stem + ".png"),
                "disease": self.data_root / "masks" / "early_blight" / "disease" / (Path(fname).stem + ".png")
            })

        # Tambahkan sampel healthy (tanpa mask diease)
        for fname in splits["healthy"][split]:
            self.samples.append({
                "class": "healthy",
                "image": self.data_root / "raw" / "Tomato___healthy" / fname,
                "leaf": self.data_root / "masks" / "healthy" / "leaf" / (Path(fname).stem + ".png"),
                "disease": None
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Baca gambar asli
        img = load_image(sample["image"])
        h, w, _ = img.shape

        # Baca mask 
        leaf_mask = load_mask(sample["leaf"], (h, w))
        disease_mask = (
            load_mask(sample["disease"], (h, w))
            if sample["class"] == "early_blight"
            else np.zeros((h, w), dtype=np.uint8)
        )

        # Resize gambar dan mask ke ukuran target
        img = cv2.resize(img, (self.image_size, self.image_size))
        leaf_mask = cv2.resize(leaf_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        disease_mask = cv2.resize(disease_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Terapkan augmentasi jika ada
        if self.transform is not None:
            augmented = self.transform(
                image=img,
                masks=[leaf_mask, disease_mask]
            )
            img, (leaf_mask, disease_mask) = augmented["image"], augmented["masks"]

        # Konversi ke tensor PyTorch dan normalisasi
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        leaf_mask = torch.from_numpy(leaf_mask).unsqueeze(0).float()
        disease_mask = torch.from_numpy(disease_mask).unsqueeze(0).float()

        # Metadata tambahan (opsional, untuk logging/debug)
        meta = {
            "class": sample["class"],
            "filename": sample["image"].name
        }

        return img, leaf_mask, disease_mask, meta