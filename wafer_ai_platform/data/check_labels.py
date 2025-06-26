import os
from collections import Counter

raw_dir = "wafer_ai_platform/data/raw"
labels = []
for fname in os.listdir(raw_dir):
    if fname.endswith('.png'):
        label = int(fname.split('_')[-1].split('.')[0])
        labels.append(label)
print("Label distribution in data/raw:", Counter(labels))
