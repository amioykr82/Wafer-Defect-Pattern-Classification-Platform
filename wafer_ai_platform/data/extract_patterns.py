import pandas as pd
import numpy as np
import cv2
import os
from collections import defaultdict

# Map string labels to class indices
LABEL_MAP = {
    'Center': 0,
    'Donut': 1,
    'Edge-Loc': 2,
    'Edge-Ring': 3,
    'Loc': 4,
    'Random': 5,
    'Scratch': 6,
    'Near-full': 7
}

PKL_PATH = "WM811K.pkl"
OUT_DIR = "wafer_ai_platform/data/patterns"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_pickle(PKL_PATH)
print('Loaded pickle. Total rows:', len(df))

# Only keep rows with string label in LABEL_MAP
pattern_rows = df[df['failureType'].apply(lambda x: isinstance(x, str) and x in LABEL_MAP)]
print('Rows with pattern labels:', len(pattern_rows))
print('Class counts:')
print(pattern_rows['failureType'].value_counts())

# Group by label for stratified split
label_indices = defaultdict(list)
for idx, row in pattern_rows.iterrows():
    label = row['failureType']
    label_indices[label].append(idx)

# Split as per user-provided counts (update as needed for your dataset)
splits = {'train': {}, 'val': {}, 'test': {}}
counts = {
    'train': {'Center':2598, 'Donut':326, 'Edge-Loc':3081, 'Edge-Ring':5873, 'Loc':2106, 'Random':516, 'Scratch':719, 'Near-full':97},
    'val':   {'Center':640,  'Donut':78,  'Edge-Loc':779,  'Edge-Ring':1426, 'Loc':571,  'Random':124, 'Scratch':186, 'Near-full':19},
    'test':  {'Center':1056, 'Donut':151, 'Edge-Loc':1329, 'Edge-Ring':2381, 'Loc':916,  'Random':226, 'Scratch':288, 'Near-full':33}
}

used = {k: set() for k in LABEL_MAP}
for split in ['train', 'val', 'test']:
    for label in LABEL_MAP:
        idxs = [i for i in label_indices[label] if i not in used[label]]
        chosen = idxs[:counts[split][label]]
        splits[split][label] = chosen
        used[label].update(chosen)

# Save images for each split
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(OUT_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    for label in LABEL_MAP:
        class_idx = LABEL_MAP[label]
        for idx in splits[split][label]:
            row = df.loc[idx]
            wafer_map = row['waferMap']
            img = np.array(wafer_map, dtype=np.uint8)
            # Map values for visibility (0->0, 1->128, 2->255)
            img_vis = np.zeros_like(img, dtype=np.uint8)
            img_vis[img == 0] = 0
            img_vis[img == 1] = 128
            img_vis[img == 2] = 255
            img_vis = cv2.resize(img_vis, (64, 64), interpolation=cv2.INTER_NEAREST)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            fname = f"{row['lotName']}_{row['waferIndex']}_{class_idx}.png"
            cv2.imwrite(os.path.join(split_dir, fname), img_vis)
print("Pattern images extracted and split into train/val/test.")
