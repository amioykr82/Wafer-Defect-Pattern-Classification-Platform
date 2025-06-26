import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

CLASS_NAMES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full'
]

class WaferDataset(Dataset):
    def __init__(self, data_dir, files=None):
        if files is not None:
            self.files = files
        else:
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.data_dir, fname))
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        img = img.transpose(2, 0, 1) / 255.0  # CHW
        label = int(fname.split('_')[-1].split('.')[0])
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 8)  # 8 classes
        )
    def forward(self, x):
        return self.fc(self.conv(x))

if __name__ == "__main__":
    train_dir = "wafer_ai_platform/data/patterns/train"
    val_dir = "wafer_ai_platform/data/patterns/val"
    test_dir = "wafer_ai_platform/data/patterns/test"

    train_files = [f for f in os.listdir(train_dir) if f.endswith('.png')]
    val_files = [f for f in os.listdir(val_dir) if f.endswith('.png')]
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]

    print("Class balance in training set:", Counter([int(f.split('_')[-1].split('.')[0]) for f in train_files]))
    print("Class balance in val set:", Counter([int(f.split('_')[-1].split('.')[0]) for f in val_files]))
    print("Class balance in test set:", Counter([int(f.split('_')[-1].split('.')[0]) for f in test_files]))

    train_ds = WaferDataset(train_dir, train_files)
    val_ds = WaferDataset(val_dir, val_files)
    test_ds = WaferDataset(test_dir, test_files)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = DeeperCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Evaluate on validation set
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs)
            pred = out.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(pred.tolist())
    print("Validation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Evaluate on test set
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs)
            pred = out.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(pred.tolist())
    print("Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    print("Test Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Export to ONNX
    dummy = torch.randn(1, 3, 64, 64)
    onnx_path = "wafer_ai_platform/models/inference_model.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['output'])
