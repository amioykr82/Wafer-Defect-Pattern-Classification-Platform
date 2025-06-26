import os

raw_dir = "wafer_ai_platform/data/raw"
no_defect = [f for f in os.listdir(raw_dir) if f.endswith('_0.png')]
print("No defect images:", no_defect[:10])  # Show first 10
print("Total no defect images:", len(no_defect))
