import os
import shutil
import re

# Source and destination directories
src_dir = os.path.join('data', 'patterns', 'test')
dst_dir = 'uploads'
os.makedirs(dst_dir, exist_ok=True)

# Prepare a dict to collect images for each class 0-7
class_images = {str(i): [] for i in range(8)}

# Regex to extract class from filename (e.g., ..._<class>.png)
pattern = re.compile(r'_(\d+)\.png$')

# Scan all files and group by class
for fname in os.listdir(src_dir):
    match = pattern.search(fname)
    if match:
        cls = match.group(1)
        if cls in class_images and len(class_images[cls]) < 2:
            class_images[cls].append(fname)

# Copy 2 images from each class to uploads/
for cls, files in class_images.items():
    for fname in files:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, f'class{cls}_{fname}')
        shutil.copy(src, dst)
        print(f'Copied {src} to {dst}')
