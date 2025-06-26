import onnxruntime as ort
import cv2
import numpy as np
import os
import torch

onnx_path = "wafer_ai_platform/models/inference_model.onnx"
session = ort.InferenceSession(onnx_path)

def predict(img_path):
    img = cv2.imread(img_path)
    img = img.transpose(2, 0, 1) / 255.0
    img = img.astype(np.float32)[None]
    out = session.run(None, {'input': img})[0]
    prob = torch.softmax(torch.tensor(out), dim=1)
    pred = prob.argmax(dim=1).item()
    conf = prob.max().item()
    return pred, conf

raw_dir = "wafer_ai_platform/data/raw"
no_defect = [f for f in os.listdir(raw_dir) if f.endswith('_0.png')]

correct = 0
for fname in no_defect[:100]:  # Check first 100 for speed
    pred, conf = predict(os.path.join(raw_dir, fname))
    print(f"{fname}: Predicted {'No Defect' if pred == 0 else 'Defect'} (Confidence: {conf:.2f})")
    if pred == 0:
        correct += 1
print(f"\nModel classified {correct}/100 no-defect images correctly as 'No Defect'. Accuracy: {correct}%")
