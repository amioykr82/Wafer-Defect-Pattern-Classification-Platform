from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
import os
import onnxruntime as ort
import torch
import shutil

app = FastAPI()

# Mount static directory for CSS
app.mount("/static", StaticFiles(directory="wafer_ai_platform/static"), name="static")

MODEL_PATH = "wafer_ai_platform/models/inference_model.onnx"
session = ort.InferenceSession(MODEL_PATH)
UPLOAD_DIR = "wafer_ai_platform/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASS_NAMES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full'
]

def predict(img_path):
    img = cv2.imread(img_path)
    print(f"[DEBUG] Loaded image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    img = img.transpose(2, 0, 1) / 255.0
    img = img.astype(np.float32)[None]
    out = session.run(None, {'input': img})[0]
    print(f"[DEBUG] Raw logits: {out}")
    prob = torch.softmax(torch.tensor(out), dim=1)
    print(f"[DEBUG] Softmax probabilities: {prob}")
    pred = prob.argmax(dim=1).item()
    conf = prob.max().item()
    return pred, conf

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    html = f"""
    <html>
    <head>
        <title>Wafer Map Pattern Classification System</title>
        <link rel="stylesheet" href="/static/style.css">
        <style>
        /* Extra inline for max compatibility */
        .container {{ box-sizing: border-box; }}
        </style>
    </head>
    <body>
        <div class='header-bar'><span class='header-title'>Wafer Map Pattern Classification System</span></div>
        <div class='main-wrapper'>
        <div class="container">
            <div class="desc">
                <p class='overview'>This system uses a deep learning model trained on the <b>WM811K wafer map dataset</b> to classify uploaded wafer images into <b>8 defect pattern classes</b>. The model is a custom deep CNN trained for multi-class pattern recognition.</p>
                <ul class='project-details'>
                    <li><b>Dataset:</b> WM811K (Kaggle, 800k+ samples)</li>
                    <li><b>Model:</b> Deep CNN, ONNX export</li>
                    <li><b>Input:</b> 64x64 wafer map PNG</li>
                    <li><b>Classes:</b> 
                        <div class='class-label-list'>
                        {''.join([f"<span class='class-badge {c.replace(' ','-')}'>{c}</span>" for c in CLASS_NAMES])}
                        </div>
                    </li>
                </ul>
            </div>
            <form action="/inspect" method="post" enctype="multipart/form-data">
                <label for="file">Upload Wafer Image (PNG):</label><br>
                <input type="file" id="file" name="file" accept="image/png" required><br><br>
                <button type="submit" class="inspect-btn">Inspect</button>
            </form>
            <div class="footer">&copy; 2025 Wafer AI Platform</div>
        </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/inspect", response_class=HTMLResponse)
def inspect(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    pred, conf = predict(file_path)
    label = CLASS_NAMES[pred]
    # Convert image to base64 for display
    import base64
    with open(file_path, "rb") as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode()
    html = f"""
    <html>
    <head>
        <title>Wafer Map Pattern Classification System</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class='main-wrapper'>
        <div class="container">
            <div class="result-card">
                <h2>Inspection Result</h2>
                <img src="data:image/png;base64,{img_b64}" class="wafer-img"/>
                <div class="result-details">
                    <span class="result-label">Predicted Pattern:</span>
                    <span class="class-badge {label.replace(' ','-')}">{label}</span>
                </div>
                <div class="result-details">
                    <span class="result-label">Confidence Score:</span>
                    <span class="confidence-score">{conf:.2%}</span>
                </div>
                <a href="/" class="inspect-another">&#8592; Inspect Another</a>
            </div>
            <div class="footer">&copy; 2025 Wafer AI Platform</div>
        </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# To run: uvicorn wafer_ai_platform.demo_ui:app --reload --port 8080
