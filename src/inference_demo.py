import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.models import UNet

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
IMAGE_SIZE = 256

# Load models
leaf_model = UNet(in_channels=3, out_channels=1).to(DEVICE)
disease_model = UNet(in_channels=3, out_channels=1).to(DEVICE)

leaf_model.load_state_dict(
    torch.load(CHECKPOINT_DIR / "leaf_best.pth", map_location=DEVICE)
)
disease_model.load_state_dict(
    torch.load(CHECKPOINT_DIR / "disease_best.pth", map_location=DEVICE)
)

leaf_model.eval()
disease_model.eval()

# Preprocess image
def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img

# Predict masks
@torch.no_grad()
def predict(img_tensor):
    img = img_tensor.unsqueeze(0).to(DEVICE)

    leaf_pred = torch.sigmoid(leaf_model(img)) > 0.5
    img_masked = img * leaf_pred

    disease_pred = torch.sigmoid(disease_model(img_masked)) > 0.5

    return (
        leaf_pred.squeeze().cpu(),
        disease_pred.squeeze().cpu()
    )

# Compute severity
def compute_severity(leaf_mask, disease_mask):
    leaf_area = leaf_mask.sum()
    disease_area = disease_mask.sum()
    return (disease_area / leaf_area).item() if leaf_area > 0 else 0.0


# Visualization
def visualize(img, leaf_mask, disease_mask, severity, title="Inference Result"):
    img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.imshow(leaf_mask, alpha=0.4, cmap="Greens")
    plt.title("Pred Leaf")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.imshow(disease_mask, alpha=0.4, cmap="Reds")
    plt.title("Pred Disease")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.text(
        0.5, 0.5,
        f"Severity:\n{severity:.3f}",
        fontsize=18,
        ha="center",
        va="center",
        transform=plt.gca().transAxes
    )
    plt.axis("off")

    plt.suptitle(title)
    plt.show()

# Main
if __name__ == "__main__":
    # GANTI PATH GAMBAR DI SINI
    image_path = PROJECT_ROOT / "data" / "raw" / "Tomato___Early_blight" / "image (127).JPG"

    img = preprocess_image(image_path)
    leaf_mask, disease_mask = predict(img)
    severity = compute_severity(leaf_mask, disease_mask)

    visualize(img, leaf_mask, disease_mask, severity, title=image_path.name)