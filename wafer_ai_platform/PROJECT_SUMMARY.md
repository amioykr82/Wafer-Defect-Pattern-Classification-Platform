# Wafer Map Pattern Classification Platform

This repository contains an end-to-end deep learning pipeline for classifying defect patterns in semiconductor wafer maps using the WM811K dataset. It includes data preprocessing, model training, evaluation, ONNX export, and a FastAPI-based web UI for real-time inference.

## Main Components
- **Data Preprocessing:** Scripts to extract, clean, and split wafer map images.
- **Model Training:** PyTorch code for a deep CNN, with evaluation and ONNX export.
- **Inference:** ONNX-based inference for batch and UI predictions.
- **Web UI:** FastAPI app for uploading and classifying wafer images interactively.

## Quick Start
1. Clone the repository.
2. Download the WM811K dataset and place `WM811K.pkl` in `wafer_ai_platform/data/`.
3. Install dependencies: `pip install -r wafer_ai_platform/requirements.txt`
4. Run data extraction: `python wafer_ai_platform/data/extract_patterns.py`
5. Train the model: `python wafer_ai_platform/models/train_model.py`
6. Launch the UI: `python wafer_ai_platform/demo_ui.py` (or use Uvicorn for production)

## Directory Structure
- `wafer_ai_platform/data/` — Data scripts and extracted images
- `wafer_ai_platform/models/` — Model training and inference scripts
- `wafer_ai_platform/static/` — CSS for the web UI
- `wafer_ai_platform/uploads/` — Temporary uploaded images for UI
- `wafer_ai_platform/demo_ui.py` — FastAPI web app
- `wafer_ai_platform/requirements.txt` — Python dependencies

## Results
See the `README.md` for latest validation accuracy, per-class metrics, and confusion matrix.

## License
MIT
