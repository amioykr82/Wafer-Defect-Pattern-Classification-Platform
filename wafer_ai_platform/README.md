# Wafer Defect & Pattern Classification Platform

## Project Overview
This project implements a complete, end-to-end AI pipeline for automatic classification of defect patterns in semiconductor wafer maps. The system loads real wafer map data, preprocesses it, trains a deep learning model, and performs inference to classify wafers into multiple defect pattern classes. The pipeline is designed for clarity, reproducibility, and executive demonstration.

---

## Dataset Description
- **Source:** [WM811K Wafer Map Dataset (Kaggle)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- **Format:** Provided as a pickled pandas DataFrame (`WM811K.pkl`)
- **Fields Used:**
  - `waferMap`: 2D array representing the wafer map (defect pattern)
  - `failureType`: String label, e.g., 'Center', 'Donut', etc. (use 'none' for no defect)
  - `lotName`, `waferIndex`: Used for unique image naming
- **Class Mapping:**
  - The main defect pattern classes are: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full
  - These string labels are mapped to integer class indices for model training.

---

## Stepwise Algorithm & Pipeline

### 1. Pattern Data Extraction & Preprocessing
- **Script:** `data/extract_patterns.py`
- Loads `WM811K.pkl`, filters rows with valid string-based pattern labels, maps them to class indices, and ignores rows with `failureType` as 'none' or not in the class map.
- Extracts each wafer map, rescales to 64x64, maps values for visibility, and saves as PNG in `data/patterns/<split>/`.
- Images are named as `<lotName>_<waferIndex>_<class_idx>.png` where `class_idx` is the mapped integer for the pattern class.
- Data is split into train, validation, and test sets with stratified sampling.

### 2. Data Inspection
- **Script:** `data/check_pattern_labels.py`
- Counts and prints the number of images per pattern class to verify class balance.

### 3. Model Training & Evaluation
- **Script:** `models/train_model.py`
- Loads images and labels from `data/patterns/<split>/` using a custom PyTorch `Dataset`.
- Uses a deeper CNN with multiple convolutional layers and dropout for robust feature extraction.
- Trains on the training set, validates on the validation set, and evaluates on the test set.
- Prints training loss, validation accuracy, per-class accuracy, and confusion matrix.
- Exports the trained model to ONNX format (`models/inference_model.onnx`).

### 4. Inference
- **Script:** `models/infer.py`
- Loads the ONNX model and runs inference on sample images from `data/patterns/test/`.
- Prints predicted class and confidence for each image.

### 5. UI Demo
- **Script:** `demo_ui.py`
- FastAPI web app for uploading and inspecting wafer images in real time using the trained model.

---

## Results (Deeper CNN, Multi-Class, 20 Epochs)
- **Validation Accuracy:** 89%
- **Test Accuracy:** 82%
- **Per-Class Accuracy (Validation Recall):**
  - Center: 99%
  - Donut: 74%
  - Edge-Loc: 90%
  - Edge-Ring: 98%
  - Loc: 63%
  - Random: 80%
  - Scratch: 73%
  - Near-full: 84%

**Validation Confusion Matrix:**

|            | Center | Donut | Edge-Loc | Edge-Ring | Loc | Random | Scratch | Near-full |
|------------|--------|-------|----------|-----------|-----|--------|---------|-----------|
| **Center** |  638   |   0   |    0     |     0     |  2  |   0    |    0    |     0     |
| **Donut**  |   1    |  58   |    1     |     0     | 16  |   1    |    1    |     0     |
| **Edge-Loc**|  3    |   0   |  702     |    19     | 31  |   0    |   24    |     0     |
| **Edge-Ring**| 0    |   0   |    2     |   1404    |  0  |   0    |   20    |     0     |
| **Loc**    |  11    |   0   |   69     |     0     | 359 |   2    |  130    |     0     |
| **Random** |   3    |   1   |   11     |     0     |  9  |   99   |    0    |     1     |
| **Scratch**|   0    |   0   |    9     |     1     | 41  |   0    |   135   |     0     |
| **Near-full**| 0    |   1   |    0     |     0     |  0  |   2    |    0    |    16     |

- **Precision, Recall, F1-score for each class are printed in the training logs.**

---

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r wafer_ai_platform/requirements.txt
   ```
2. **Extract pattern images from the dataset:**
   ```bash
   python wafer_ai_platform/data/extract_patterns.py
   ```
3. **Check class distribution (optional):**
   ```bash
   python wafer_ai_platform/data/check_pattern_labels.py
   ```
4. **Train the model:**
   ```bash
   python wafer_ai_platform/models/train_model.py
   ```
5. **Run inference:**
   ```bash
   python wafer_ai_platform/models/infer.py
   ```
6. **Launch the demo UI:**
   ```bash
   uvicorn wafer_ai_platform.demo_ui:app --reload --port 8080
   ```
   Then open [http://localhost:8080](http://localhost:8080) in your browser.

---

## Files Used in This Project
- `data/extract_patterns.py` — Extracts and saves wafer pattern images with string-based class labels mapped to indices
- `data/check_pattern_labels.py` — Checks class balance in the extracted pattern images
- `models/train_model.py` — Trains the deeper CNN for multi-class pattern classification and exports the ONNX model
- `models/infer.py` — Runs inference on sample pattern images
- `demo_ui.py` — FastAPI web UI for real-time inspection
- `WM811K.pkl` — The wafer map dataset (download from Kaggle)

---

## Notes
- The model now performs multi-class pattern classification using string-based class labels mapped to indices.
- All scripts are self-contained and can be run in sequence for a full demo.
- For best results, run in a Python 3.8+ environment with sufficient RAM and disk space.
