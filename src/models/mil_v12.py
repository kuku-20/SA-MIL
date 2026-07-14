


# Project root (auto-detected)

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

"""
Structure-aware MIL v12 - 熵驱动显著性采样版 (Entropy-Driven Saliency Sampling)
================================================================================

核心改进：
1. 【显著性切窗】对于超出物理覆盖极限的大文件，采用基于 B/G 通道的信息熵和结构异常显著性评分。
2. 【一维 NMS】使用 1D Non-Maximum Suppression 防止高危窗口重叠，将窗口“砸向”恶意载荷最密集的区域。
3. 【时序排序】智能采样后，将窗口按物理位置重新排序，保持 Transformer 输入的时序语义。
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
SEEDS = [123] # 集成用的随机种子
MAX_EPOCHS = 30
PATIENCE = 10

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

                if 'image_path' in row and row['image_path']:
                    img_path = self.images_dir / row['image_path']
                else:
                    img_path = self.images_dir / family / f"{sha256}.png"

                if img_path.exists():
                    self.samples.append((str(img_path), self.class_to_idx[family]))

        self.class_counts = Counter([s[1] for s in self.samples])
        self.global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
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
        # Window 0: 依然是全局视野
        windows = [self.global_resize(img)] 
        
        img_resized = img
        new_h = h
        
        if w != FIXED_WIDTH:
            new_h = int(h * FIXED_WIDTH / w)
            img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)

        # 正常小文件，直接 Padding
        if new_h <= WINDOW_H:
            canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
            canvas.paste(img_resized, (0, 0))
            while len(windows) < MAX_WINDOWS:
                windows.append(canvas)
            return windows

        local_max_windows = MAX_WINDOWS - 1 # 11
        max_cover_h = WINDOW_STRIDE * (local_max_windows - 1) + WINDOW_H

        # 如果是常规文件（能被顺序覆盖），为了速度保持顺序滑动
        if new_h <= max_cover_h:
            for i in range(local_max_windows):
                top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
                windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            return windows

        # =================================================================
        # 1% 超大文件：启动【熵与结构驱动的显著性采样】
        # =================================================================
        img_np = np.array(img_resized)  # [H, W, 3]
        
        # G通道 (索引1) 是结构异常，B通道 (索引2) 是信息熵
        # 计算每一行的“显著性得分”
        row_scores = np.sum(img_np[:, :, 1:3], axis=(1, 2)) 
        
        search_stride = 32
        candidate_windows = []
        
        # 滑动计算所有候选窗口的总得分
        for top in range(0, new_h - WINDOW_H + 1, search_stride):
            window_score = np.sum(row_scores[top : top + WINDOW_H])
            candidate_windows.append((window_score, top))
            
        # 按得分从高到低排序，最危险的窗口排在前面
        candidate_windows.sort(key=lambda x: x[0], reverse=True)
        
        # 一维非极大值抑制 (1D-NMS)，防止窗口过度重叠
        selected_tops = []
        min_distance = WINDOW_H // 2  # 允许最多 50% 的重叠
        
        # 强制把文件最头部（PE结构）加入，包含重要元数据
        selected_tops.append(0)
        
        for score, top in candidate_windows:
            if len(selected_tops) >= local_max_windows:
                break
            if all(abs(top - sel_top) >= min_distance for sel_top in selected_tops):
                selected_tops.append(top)
                
        # 补齐：如果因为全是0没选满，用最尾部的区域补齐
        while len(selected_tops) < local_max_windows:
            fallback_top = new_h - WINDOW_H - (local_max_windows - len(selected_tops)) * search_stride
            fallback_top = max(0, fallback_top)
            selected_tops.append(fallback_top)

        # 重新按 Y 轴物理顺序排序（保证输入 Transformer 的时序语义正确）
        selected_tops.sort()
        
        for top in selected_tops[:local_max_windows]:
            windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))

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
        
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss/len(loader), acc, f1

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
            logits1 = m(windows, masks)
            windows_flip = torch.flip(windows, dims=[-1])
            logits2 = m(windows_flip, masks)
            probs = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1)) / 2.0
            batch_probs += probs
            
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
    print(f"Device: {device}\n" + "="*60 + "\nStructure-aware MIL v12 (Entropy-Driven Saliency Sampling)\n" + "="*60)
    print(f"Classes ({NUM_CLASSES}): {families}")

    train_dataset = MalwareDataset(train_csv, images_dir, CLASS_TO_IDX, train=True)
    val_dataset = MalwareDataset(val_csv, images_dir, CLASS_TO_IDX, train=False)
    test_dataset = MalwareDataset(test_csv, images_dir, CLASS_TO_IDX, train=False)
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    saved_models = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Training Model with Seed: {seed}")
        set_seed(seed)
        
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
        model_name = f"best_mil_v12_seed{seed}.pth"
        epoch_list, train_f1_list, val_f1_list = [], [], []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch:3d} | "
                f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )
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

        plot_training_curves(
            epoch_list, train_f1_list, val_f1_list,
            save_path=f"training_curve_seed{seed}.png"
        )
        saved_models.append(model_name)
        
    print("\n" + "="*60 + "\nFINAL TESTING (Ensemble 3 Models + TTA)\n" + "="*60)
    acc_tta, f1_tta, preds, labels = evaluate_ensemble_tta(saved_models, test_loader, device)
    
    print(f"\nFinal Test Accuracy: {acc_tta:.4f}")
    print(f"Final Test Macro F1: {f1_tta:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)])}")

    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png")

if __name__ == "__main__":
    main()
