import pandas as pd
import numpy as np
import os
from zipfile import ZipFile

def extract_wm811k(zip_path, output_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

if __name__ == "__main__":
    extract_wm811k("data/wm811k-wafer-map.zip", "data/raw")
