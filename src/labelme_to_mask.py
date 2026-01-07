import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Path Setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"

EARLY_DIR = DATA_ROOT / "raw" / "Tomato___Early_blight"
HEALTHY_DIR = DATA_ROOT / "raw" / "Tomato___healthy"

LEAF_EARLY_OUT = DATA_ROOT / "masks" / "early_blight" / "leaf"
DISEASE_EARLY_OUT = DATA_ROOT / "masks" / "early_blight" / "disease"

LEAF_HEALTHY_OUT = DATA_ROOT / "masks" / "healthy" / "leaf"

LEAF_EARLY_OUT.mkdir(parents=True, exist_ok=True)
DISEASE_EARLY_OUT.mkdir(parents=True, exist_ok=True)
LEAF_HEALTHY_OUT.mkdir(parents=True, exist_ok=True)

# Load annotation list
ANNOT_LIST_PATH = DATA_ROOT / "annotation_list.json"

with open(ANNOT_LIST_PATH, "r") as f:
    annotation_list = json.load(f)

early_files = set(annotation_list["early_blight"])
healthy_files = set(annotation_list["healthy"])

print(f"Annotated Early Blight : {len(early_files)}")
print(f"Annotated Healthy     : {len(healthy_files)}")

# LabelMe -> mask
def labelme_to_masks(json_path, image_shape):
    with open(json_path, "r") as f:
        data = json.load(f)

    leaf_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    disease_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = shape["label"].lower()
        points = np.array(shape["points"], dtype=np.int32)

        if label == "leaf":
            cv2.fillPoly(leaf_mask, [points], 255)
        elif label == "disease":
            cv2.fillPoly(disease_mask, [points], 255)

    return leaf_mask, disease_mask

# Process Early Blight
for fname in tqdm(sorted(early_files), desc="Processing Early Blight"):
    img_path = EARLY_DIR / fname
    json_path = img_path.with_suffix(".json")

    if not img_path.exists():
        print(f"Missing image: {fname}")
        continue
    if not json_path.exists():
        print(f"Missing annotation: {fname}")
        continue

    img = cv2.imread(str(img_path))
    leaf_mask, disease_mask = labelme_to_masks(json_path, img.shape)

    out_name = img_path.stem + ".png"
    cv2.imwrite(str(LEAF_EARLY_OUT / out_name), leaf_mask)
    cv2.imwrite(str(DISEASE_EARLY_OUT / out_name), disease_mask)


# Process Healthy
for fname in tqdm(sorted(healthy_files), desc="Processing Healthy"):
    img_path = HEALTHY_DIR / fname
    json_path = img_path.with_suffix(".json")

    if not img_path.exists():
        print(f"Missing image: {fname}")
        continue
    if not json_path.exists():
        print(f"Missing annotation: {fname}")
        continue

    img = cv2.imread(str(img_path))
    leaf_mask, _ = labelme_to_masks(json_path, img.shape)

    out_name = img_path.stem + ".png"
    cv2.imwrite(str(LEAF_HEALTHY_OUT / out_name), leaf_mask)