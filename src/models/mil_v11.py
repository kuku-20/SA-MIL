

# Project root (auto-detected)

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

"""
Structure-aware MIL v11 - 终极正则化打榜版 (Ensemble + RandAugment + Label Smoothing)
================================================================================

核心改进：
1. 【增强升级】引入 RandAugment，提供谷歌级的强大组合数据增强。
2. 【标签平滑】CrossEntropyLoss 引入 label_smoothing=0.1，缓解过拟合，提升泛化。
3. 【强力正则】分类头 Dropout 提高至 0.5。
"""

import os
import csv
import math
import random
from PIL import Image
import numpy as np
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 引入绘图工具
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.training.visualize import plot_confusion_matrix, plot_training_curves

# ===============================
# Config
# ===============================
CLASS_TO_IDX = {}
IDX_TO_CLASS = {}
NUM_CLASSES = 0

FIXED_WIDTH   = 256
WINDOW_H      = 224
WINDOW_STRIDE = 112
MAX_WINDOWS   = 12

HIDDEN_DIM = 512
SEEDS = [42, 123, 456] # 集成用的随机种子
MAX_EPOCHS = 50
PATIENCE = 15

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===============================
# Dataset
# ===============================
class MalwareDataset(Dataset):
    def __init__(self, csv_path, images_dir, class_to_idx, train=True):
        self.images_dir = Path(images_dir)
        self.samples = []
        self.train = train
        self.class_to_idx = class_to_idx

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sha256 = row['sha256']
                family = row['family']
                if family not in self.class_to_idx:
                    continue

                # 优先使用新 CSV 的 image_path 列（相对 images_dir）
                if 'image_path' in row and row['image_path']:
                    img_path = self.images_dir / row['image_path']
                else:
                    # 兼容旧格式：images_dir/family/sha256.png
                    img_path = self.images_dir / family / f"{sha256}.png"

                if img_path.exists():
                    self.samples.append((str(img_path), self.class_to_idx[family]))

        self.class_counts = Counter([s[1] for s in self.samples])
        self.global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # [V11 改进] 引入 RandAugment 强大的组合数据增强
        if train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandAugment(num_ops=2, magnitude=9), # 替换掉原有的 ColorJitter
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)) # Cutout
            ])
        else:
            self.augment = None
            self.eval_transform = transforms.ToTensor()

    def __len__(self): return len(self.samples)
    
    def get_sample_weights(self):
        total = sum(self.class_counts.values())
        return [total / (NUM_CLASSES * self.class_counts[label]) for _, label in self.samples]

    def _extract_windows(self, img):
        w, h = img.size
        windows = [self.global_resize(img)]
        
        if w != FIXED_WIDTH:
            new_h = int(h * FIXED_WIDTH / w)
            img = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)
            h = new_h

        if h <= WINDOW_H:
            canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
            canvas.paste(img, (0, 0))
            windows.append(canvas)
        else:
            num_steps = min(math.ceil((h - WINDOW_H) / WINDOW_STRIDE) + 1, MAX_WINDOWS - 1)
            for i in range(num_steps):
                top = min(i * WINDOW_STRIDE, h - WINDOW_H)
                windows.append(img.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
                if top + WINDOW_H >= h: break

        return windows

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        windows = self._extract_windows(img)

        if self.train:
            tensors = [self.normalize(self.augment(w)) for w in windows]
        else:
            tensors = [self.normalize(self.eval_transform(w)) for w in windows]
            
        return torch.stack(tensors), label, len(windows)

def collate_fn(batch):
    win_list, labels, nw_list = zip(*batch)
    B = len(win_list)
    max_wins = max(nw_list)
    _, c, h, w = win_list[0].shape

    windows = torch.zeros(B, max_wins, c, h, w)
    masks = torch.zeros(B, max_wins, dtype=torch.bool)

    for i, (wins, nw) in enumerate(zip(win_list, nw_list)):
        windows[i, :nw] = wins
        masks[i, :nw] = True

    return windows, torch.tensor(labels), masks

# ===============================
# Model (MPS 安全版)
# ===============================
class SafeTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask):
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ResNetMIL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.avgpool = backbone.avgpool
        
        for i, child in enumerate(self.features.children()):
            if i < 6: 
                for p in child.parameters(): p.requires_grad = False
                
        self.proj = nn.Sequential(
            nn.Linear(2048, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_WINDOWS, HIDDEN_DIM) * 0.02)
        
        self.transformer = SafeTransformerLayer(
            d_model=HIDDEN_DIM, nhead=4, dim_feedforward=HIDDEN_DIM*2, dropout=0.1
        )
        
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # [V11 改进] 提升 Dropout 至 0.5
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_DIM, num_classes)
        )

    def forward(self, windows, masks):
        B, N, c, h, w = windows.shape
        x = self.features(windows.view(B * N, c, h, w))
        x = self.avgpool(x).flatten(1)
        x = self.proj(x).view(B, N, -1)
        
        x = x + self.pos_embed[:, :N, :]
        padding_mask = ~masks
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        attn_scores = self.attention(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~masks, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)
        return self.classifier(pooled)

# ===============================
# Training Utilities
# ===============================
def train_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for windows, labels, masks in tqdm(loader, desc="Train", leave=False):
        windows, labels, masks = windows.to(device), labels.to(device), masks.to(device)
        
        optimizer.zero_grad()
        logits = model(windows, masks)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return total_loss/len(loader), f1_score(all_labels, all_preds, average='macro')

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for windows, labels, masks in tqdm(loader, desc="Eval", leave=False):
        windows, labels, masks = windows.to(device), labels.to(device), masks.to(device)
        
        logits = model(windows, masks)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss/len(loader), acc, f1, all_preds, all_labels

@torch.no_grad()
def evaluate_ensemble_tta(model_paths, loader, device):
    models_list = []
    print("\nLoading models for ensemble...")
    for path in model_paths:
        m = ResNetMIL(NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(path))
        m.eval()
        models_list.append(m)

    all_preds, all_labels = [], []
    for windows, labels, masks in tqdm(loader, desc="Ensemble+TTA Eval"):
        windows, labels, masks = windows.to(device), labels.to(device), masks.to(device)
        
        batch_probs = torch.zeros(windows.size(0), NUM_CLASSES).to(device)
        
        for m in models_list:
            # 原始预测
            logits1 = m(windows, masks)
            # 水平翻转预测 (TTA)
            windows_flip = torch.flip(windows, dims=[-1])
            logits2 = m(windows_flip, masks)
            
            # 平均当前模型的 TTA 概率
            probs = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1)) / 2.0
            batch_probs += probs
            
        # 平均所有模型的概率
        batch_probs /= len(models_list)
        preds = batch_probs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, all_preds, all_labels

# ===============================
# Main
# ===============================
def main():
    global CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES

images_dir = str(_PROJ_DIR / "data" / "malware_images")
train_csv = str(_PROJ_DIR / "data" / "malware_images" / "train.csv")
val_csv = str(_PROJ_DIR / "data" / "malware_images" / "val.csv")
test_csv = str(_PROJ_DIR / "data" / "malware_images" / "test.csv")

    # 动态构建类别映射（当前数据集为 6 类）
    families = set()
    for csv_path in [train_csv, val_csv, test_csv]:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                families.add(row['family'])

    families = sorted(families)
    CLASS_TO_IDX = {fam: i for i, fam in enumerate(families)}
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    NUM_CLASSES = len(CLASS_TO_IDX)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n" + "="*60 + "\nStructure-aware MIL v11 (Ensemble + RandAugment + LabelSmoothing)\n" + "="*60)
    print(f"Classes ({NUM_CLASSES}): {families}")

    train_dataset = MalwareDataset(train_csv, images_dir, CLASS_TO_IDX, train=True)
    val_dataset = MalwareDataset(val_csv, images_dir, CLASS_TO_IDX, train=False)
    test_dataset = MalwareDataset(test_csv, images_dir, CLASS_TO_IDX, train=False)
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # [V11 改进] 引入标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    saved_models = []

    # 循环训练3个不同随机种子的模型
    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Training Model with Seed: {seed}")
        set_seed(seed)
        
        # 重新初始化 sampler，保证不同 seed 采样序列不同
        sampler = WeightedRandomSampler(train_dataset.get_sample_weights(), len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn, num_workers=0)
        
        model = ResNetMIL(NUM_CLASSES).to(device)
        
        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': 2e-5},
            {'params': model.proj.parameters(), 'lr': 2e-4},
            {'params': model.transformer.parameters(), 'lr': 2e-4},
            {'params': model.attention.parameters(), 'lr': 2e-4},
            {'params': model.classifier.parameters(), 'lr': 2e-4},
            {'params': [model.pos_embed], 'lr': 2e-4}
        ], weight_decay=1e-3)
        
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[2e-5, 2e-4, 2e-4, 2e-4, 2e-4, 2e-4], 
                              total_steps=len(train_loader)*MAX_EPOCHS, pct_start=0.1)
        
        best_f1 = 0
        no_improve_epochs = 0
        model_name = f"best_mil_v11_seed{seed}.pth"
        epoch_list, train_f1_list, val_f1_list = [], [], []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

            print(f"Epoch {epoch:3d} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            epoch_list.append(epoch)
            train_f1_list.append(train_f1)
            val_f1_list.append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), model_name)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

        # 保存训练曲线图
        plot_training_curves(
            epoch_list, train_f1_list, val_f1_list,
            save_path=f"training_curve_seed{seed}.png"
        )
        saved_models.append(model_name)
        
    # 最终测试 (Ensemble + TTA)
    print("\n" + "="*60 + "\nFINAL TESTING (Ensemble 3 Models + TTA)\n" + "="*60)
    acc_tta, f1_tta, preds, labels = evaluate_ensemble_tta(saved_models, test_loader, device)
    
    print(f"\nFinal Test Accuracy: {acc_tta:.4f}")
    print(f"Final Test Macro F1: {f1_tta:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)])}")

    # 保存混淆矩阵图
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png")

if __name__ == "__main__":
    main()