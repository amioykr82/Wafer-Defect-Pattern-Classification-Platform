import os
import shutil

# List of 16 required images (2 per class, update these as needed)
required_images = [
    'lot39578_21.0_0.png', 'lot39579_1.0_0.png',  # Center
    'lot39579_23.0_0.png', 'lot39579_24.0_0.png',  # Donut
    'lot39580_1.0_0.png', 'lot39580_23.0_0.png',  # Edge-Loc
    'lot39580_24.0_0.png', 'lot39580_3.0_0.png',  # Edge-Ring
    'lot39593_11.0_0.png', 'lot39593_15.0_0.png',  # Loc
    'lot39593_17.0_0.png', 'lot39593_21.0_0.png',  # Random
    'lot39593_5.0_0.png', 'lot39594_11.0_0.png',  # Scratch
    'lot39594_15.0_0.png', 'lot39594_17.0_0.png',  # Near-full
]

src_dir = os.path.join(os.path.dirname(__file__), 'data', 'patterns', 'test')
dst_dir = os.path.join(os.path.dirname(__file__), 'uploads')

os.makedirs(dst_dir, exist_ok=True)

for img in required_images:
    src = os.path.join(src_dir, img)
    dst = os.path.join(dst_dir, img)
    try:
        shutil.copy2(src, dst)
        print(f'Copied: {img}')
    except Exception as e:
        print(f'Failed to copy {img}: {e}')

# Verify all images are present in uploads/
missing = []
for img in required_images:
    if not os.path.exists(os.path.join(dst_dir, img)):
        missing.append(img)

if missing:
    print('Missing in uploads:', missing)
else:
    print('All 16 images are present in uploads/.')
