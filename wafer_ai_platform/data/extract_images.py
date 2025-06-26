import pandas as pd
import numpy as np
import cv2
import os

# Path to the pickle file
PKL_PATH = "WM811K.pkl"
# Output directory for images
OUT_DIR = "wafer_ai_platform/data/raw"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_pickle(PKL_PATH)

# Mapping: 0->0 (background), 1->128 (normal), 2->255 (defect)
def wafer_map_to_img(arr):
    arr = np.array(arr, dtype=np.uint8)
    img = np.zeros_like(arr, dtype=np.uint8)
    img[arr == 0] = 0
    img[arr == 1] = 128
    img[arr == 2] = 255
    return img

for idx, row in df.iterrows():
    wafer_map = row['waferMap']
    failure_type = str(row['failureType'])
    label = 1 if failure_type.strip().lower() != 'none' else 0
    img = wafer_map_to_img(wafer_map)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fname = f"{row['lotName']}_{row['waferIndex']}_{label}.png"
    cv2.imwrite(os.path.join(OUT_DIR, fname), img)
    if idx % 10000 == 0:
        print(f"Processed {idx} images...")
print("Image extraction complete.")
