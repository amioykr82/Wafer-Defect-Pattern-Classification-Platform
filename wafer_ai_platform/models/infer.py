import onnxruntime as ort
import cv2
import numpy as np
import os
import torch

CLASS_NAMES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full'
]

onnx_path = "wafer_ai_platform/models/inference_model.onnx"
session = ort.InferenceSession(onnx_path)

def predict(img_path):
    img = cv2.imread(img_path)
    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    img = img.transpose(2, 0, 1) / 255.0
    img = img.astype(np.float32)[None]
    out = session.run(None, {'input': img})[0]
    print(f"Raw logits: {out}")
    prob = torch.softmax(torch.tensor(out), dim=1)
    print(f"Softmax probabilities: {prob}")
    pred = prob.argmax(dim=1).item()
    conf = prob.max().item()
    return pred, conf

# Test on a specific image
img_path = "wafer_ai_platform/data/patterns/test/lot46873_8.0_0.png"
pred, conf = predict(img_path)
print(f"{os.path.basename(img_path)}: {CLASS_NAMES[pred]} (Confidence: {conf:.2f})")

# Test on first 10 images in test set
test_dir = "wafer_ai_platform/data/patterns/test"
for fname in os.listdir(test_dir)[:10]:
    pred, conf = predict(os.path.join(test_dir, fname))
    print(f"{fname}: {CLASS_NAMES[pred]} (Confidence: {conf:.2f})")
