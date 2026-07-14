import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import warnings


# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

warnings.filterwarnings("ignore")
from tqdm import tqdm

def resolve_path(p):
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
    if not os.path.exists(p):
        raise FileNotFoundError(f"路径不存在：{p}")
    return p

_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_BASE, "..", ".."))
TRAIN_CSV = os.path.join(_PROJ_ROOT, "data", "malware_images", "train.csv")
VAL_CSV = os.path.join(_PROJ_ROOT, "data", "malware_images", "val.csv")
TEST_CSV = os.path.join(_PROJ_ROOT, "data", "malware_images", "test.csv")
_PROJ_DIR / "data" / "malware_images")  # 不加 " / "images"

class BaselineDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.data['family'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_path = str(row['image_path']).strip()
        candidates = [
            raw_path,
            os.path.join(self.img_dir, raw_path),
        ]
        if raw_path.startswith("images/"):
            candidates.append(os.path.join(self.img_dir, raw_path[len("images/"):]))
        if raw_path.startswith("/"):
            candidates.insert(0, raw_path)

        img_path = None
        for cand in candidates:
            cand = os.path.normpath(cand)
            if os.path.exists(cand):
                img_path = cand
                break

        if img_path is None:
            raise FileNotFoundError(f"图片不存在：{raw_path}（尝试路径：{candidates}）")

        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        img = img.repeat(3, 1, 1)
        label = self.class_to_idx[row['family']]
        return img, label

def main():
    BATCH_SIZE = 16
    EPOCHS = 10
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = BaselineDataset(TRAIN_CSV, IMG_DIR, transform)
    val_dataset = BaselineDataset(VAL_CSV, IMG_DIR, transform)
    test_dataset = BaselineDataset(TEST_CSV, IMG_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    NUM_CLASSES = len(train_dataset.classes)
    model = models.resnet18(pretrained=False)       # 改小模型 + 不预训练
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),                          # 加 dropout
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # 更高正则

    print(f"[INFO] dataset: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_preds, total_labels = 0.0, [], []
        with tqdm(train_loader, desc=f"Train ep{epoch}", unit="batch") as pbar:
            for batch_idx, (imgs, labels) in enumerate(pbar, 1):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                total_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                total_labels.extend(labels.cpu().numpy())

                pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        train_f1 = f1_score(total_labels, total_preds, average='macro')

        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
            val_f1 = f1_score(val_labels, val_preds, average='macro')

            with torch.no_grad():
                test_preds, test_labels = [], []
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
                test_f1 = f1_score(test_labels, test_preds, average='macro')

        print(f"Epoch {epoch:2d} | train_f1={train_f1:.4f} | val_f1={val_f1:.4f} | test_f1={test_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_baseline_resnet18.pth")
            print(f"[SAVE] Model saved at epoch {epoch} with val_f1={val_f1:.4f}")

    print("best_val_f1:", best_val_f1)

if __name__ == "__main__":
    main()
