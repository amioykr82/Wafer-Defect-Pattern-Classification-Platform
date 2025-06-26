import numpy as np
import cv2
import os
import argparse

def save_wafer_map(wafer_map, out_path):
    img = np.array(wafer_map, dtype=np.uint8)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_path, img)
    print(f"Saved wafer map image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a wafer map numpy file as a PNG image.")
    parser.add_argument('--npy', type=str, required=True, help='Path to .npy file containing wafer map (2D array)')
    parser.add_argument('--out', type=str, required=True, help='Output PNG path (e.g. wafer_ai_platform/data/raw/mywafer_1.png)')
    args = parser.parse_args()

    wafer_map = np.load(args.npy)
    save_wafer_map(wafer_map, args.out)
