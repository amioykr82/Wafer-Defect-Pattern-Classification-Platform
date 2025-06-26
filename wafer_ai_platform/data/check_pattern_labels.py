import os
from collections import Counter

for split in ['train', 'val', 'test']:
    split_dir = f"wafer_ai_platform/data/patterns/{split}"
    files = [f for f in os.listdir(split_dir) if f.endswith('.png')]
    labels = [int(f.split('_')[-1].split('.')[0]) for f in files]
    print(f"{split.title()} set class distribution:", Counter(labels))
    print(f"Total {split} images:", len(files))
