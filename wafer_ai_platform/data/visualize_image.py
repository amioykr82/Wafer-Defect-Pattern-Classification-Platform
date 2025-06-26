import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

RAW_DIR = "wafer_ai_platform/data/raw"

if __name__ == "__main__":
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.png')]
    if not files:
        print(f"No PNG images found in {RAW_DIR}")
        exit(1)
    n = min(8, len(files))
    plt.figure(figsize=(12, 6))
    for i, fname in enumerate(files[:n]):
        img = cv2.imread(os.path.join(RAW_DIR, fname), cv2.IMREAD_GRAYSCALE)
        # If the image is all zeros, try to enhance contrast for visualization
        if np.max(img) == 0:
            print(f"Warning: {fname} is all zeros.")
        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(fname)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
