# Wafer Map Pattern Classification Platform

This repository contains an end-to-end deep learning pipeline for classifying defect patterns in semiconductor wafer maps using the WM811K dataset. It includes data preprocessing, model training, evaluation, ONNX export, and a FastAPI-based web UI for real-time inference.

## Main Components
- **Data Preprocessing:** Scripts to extract, clean, and split wafer map images (in `data/`).
- **Model Training:** PyTorch code for a deep CNN, with evaluation and ONNX export (in `models/`).
- **Inference:** ONNX-based inference for batch and UI predictions (in `models/`).
- **Web UI:** FastAPI app for uploading and classifying wafer images interactively (`demo_ui.py` + `static/`).

## Quick Start
1. Clone the repository.
2. Download the WM811K dataset and place `WM811K.pkl` in the project root.
3. Install dependencies: `pip install -r requirements.txt`
4. Run data extraction: `python data/extract_patterns.py`
5. Train the model: `python models/train_model.py`
6. Launch the UI: `python demo_ui.py` (or use Uvicorn for production)

## Directory Structure
- `data/` — Data scripts (no large data or generated images)
- `models/` — Model training and inference scripts
- `static/` — CSS for the web UI
- `test_image/` — Sample images for UI testing
- `demo_ui.py` — FastAPI web app
- `requirements.txt` — Python dependencies
- `.gitignore`, `LICENSE`, `README.md`, `PROJECT_SUMMARY.md` — Repo hygiene & docs

## Results
See the `README.md` for latest validation accuracy, per-class metrics, and confusion matrix.

## License
MIT
