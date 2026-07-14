


# Project root (auto-detected)

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

"""
Channel Ablation Study (Aligned to Structure-aware MIL v12)
================================================================================
基于 `StructureAwareMIL_v12.py` 的切窗逻辑与训练流程进行通道消融。

模式：
- 'RGB': 完整方法 (Byte + Structure + Entropy)
- 'R'  : 传统基线方法 (纯灰度图, G=R, B=R)
- 'RG' : Byte + Structure
- 'RB' : Byte + Entropy
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
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
SEED_FOR_ABLATION = 42
MAX_EPOCHS = 25  
PATIENCE = 8

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===============================
# Dataset with Channel Ablation
# ===============================
class MalwareDataset(Dataset):
    def __init__(self, csv_path, images_dir, class_to_idx, train=True, channel_mode='RGB'):
        self.images_dir = Path(images_dir)
        self.samples = []
        self.train = train
        self.class_to_idx = class_to_idx
        self.channel_mode = channel_mode

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

        # 与 StructureAwareMIL_v12 保持一致的增强策略
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

    def _apply_channel_mask(self, img):
        """核心逻辑：动态通道屏蔽"""
        if self.channel_mode == 'RGB':
            return img # 完整方法，不修改
            
        img_np = np.array(img)
        
        if self.channel_mode == 'R':
            # 传统灰度图：把 G 和 B 都变成 R
            img_np[:, :, 1] = img_np[:, :, 0]
            img_np[:, :, 2] = img_np[:, :, 0]
        elif self.channel_mode == 'RG':
            # 只用结构：把 B(熵) 变成 R
            img_np[:, :, 2] = img_np[:, :, 0]
        elif self.channel_mode == 'RB':
            # 只用熵：把 G(结构) 变成 R
            img_np[:, :, 1] = img_np[:, :, 0]
            
        return Image.fromarray(img_np)

    def _extract_windows(self, img):
        # 基于 StructureAwareMIL_v12 的切窗逻辑：
        # - W0 全局窗口
        # - 小文件：Padding 补齐 MAX_WINDOWS
        # - 常规文件：顺序滑窗
        # - 超大文件：熵/结构驱动的显著性采样 + 1D-NMS
        w, h = img.size
        windows = [self.global_resize(img)]  # Window 0: 全局视野

        img_resized = img
        new_h = h
        if w != FIXED_WIDTH:
            new_h = int(h * FIXED_WIDTH / w)
            img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)

        # 正常小文件：Padding
        if new_h <= WINDOW_H:
            canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
            canvas.paste(img_resized, (0, 0))
            while len(windows) < MAX_WINDOWS:
                windows.append(canvas)
            return windows

        local_max_windows = MAX_WINDOWS - 1
        max_cover_h = WINDOW_STRIDE * (local_max_windows - 1) + WINDOW_H

        # 常规文件：顺序滑动（为了速度）
        if new_h <= max_cover_h:
            for i in range(local_max_windows):
                top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
                windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            return windows

        # 超大文件：熵/结构驱动显著性采样
        img_np = np.array(img_resized)  # [H, W, 3]
        # G通道(索引1) 结构异常，B通道(索引2) 信息熵
        row_scores = np.sum(img_np[:, :, 1:3], axis=(1, 2))

        search_stride = 32
        candidate_windows = []
        for top in range(0, new_h - WINDOW_H + 1, search_stride):
            window_score = np.sum(row_scores[top: top + WINDOW_H])
            candidate_windows.append((window_score, top))

        candidate_windows.sort(key=lambda x: x[0], reverse=True)

        selected_tops = []
        min_distance = WINDOW_H // 2  # 允许最多 50% 重叠
        selected_tops.append(0)  # 强制包含头部（PE结构/元数据）

        for score, top in candidate_windows:
            if len(selected_tops) >= local_max_windows:
                break
            if all(abs(top - sel_top) >= min_distance for sel_top in selected_tops):
                selected_tops.append(top)

        # 补齐：如果没选满，用尾部区域补齐
        while len(selected_tops) < local_max_windows:
            fallback_top = new_h - WINDOW_H - (local_max_windows - len(selected_tops)) * search_stride
            fallback_top = max(0, fallback_top)
            selected_tops.append(fallback_top)

        selected_tops.sort()  # 时序排序（保证 Transformer 语义）
        for top in selected_tops[:local_max_windows]:
            windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))

        return windows

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # 1) 应用通道掩码（影响熵/结构打分与窗口采样）
        img = self._apply_channel_mask(img)

        # 2) 提取窗口
        windows = self._extract_windows(img)

        # 3) Tensor 化 + 归一化（与 v12 对齐）
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
# Model (与之前完全相同)
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
            nn.Linear(2048, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU(), nn.Dropout(0.2)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_WINDOWS, HIDDEN_DIM) * 0.02)
        self.transformer = SafeTransformerLayer(d_model=HIDDEN_DIM, nhead=4, dim_feedforward=HIDDEN_DIM*2, dropout=0.1)
        self.attention = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(HIDDEN_DIM, num_classes))

    def forward(self, windows, masks):
        B, N, c, h, w = windows.shape
        x = self.features(windows.view(B * N, c, h, w))
        x = self.avgpool(x).flatten(1)
        x = self.proj(x).view(B, N, -1)
        x = x + self.pos_embed[:, :N, :]
        x = self.transformer(x, src_key_padding_mask=~masks)
        attn_scores = self.attention(x).squeeze(-1).masked_fill(~masks, -1e9)
        pooled = (F.softmax(attn_scores, dim=-1).unsqueeze(-1) * x).sum(dim=1)
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
    return total_loss / len(loader), acc, f1

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
    return total_loss / len(loader), acc, f1, all_preds, all_labels

# ===============================
# Main Execution
# ===============================
def main():
    global CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
    
    # === 请在此替换为你的实际路径 ===
    save_dir = Path(__file__).resolve().parent
images_dir = str(_PROJ_DIR / "data" / "malware_images")
train_csv = str(_PROJ_DIR / "data" / "malware_images" / "train.csv")
val_csv = str(_PROJ_DIR / "data" / "malware_images" / "val.csv")
test_csv = str(_PROJ_DIR / "data" / "malware_images" / "test.csv")

    families = set()
    for csv_path in [train_csv, val_csv, test_csv]:
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f): families.add(row['family'])
    CLASS_TO_IDX = {fam: i for i, fam in enumerate(sorted(families))}
    NUM_CLASSES = len(CLASS_TO_IDX)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n" + "="*60 + "\nStarting CHANNEL Ablation Study\n" + "="*60)

    # 4 种通道模式对照
    channel_modes = ['R', 'RG', 'RB', 'RGB']
    results_dict = {}

    for mode in channel_modes:
        print(f"\n▶▶▶ Running Channel Mode: [{mode}]")
        set_seed(SEED_FOR_ABLATION)
        
        train_dataset = MalwareDataset(train_csv, images_dir, CLASS_TO_IDX, train=True, channel_mode=mode)
        val_dataset = MalwareDataset(val_csv, images_dir, CLASS_TO_IDX, train=False, channel_mode=mode)
        test_dataset = MalwareDataset(test_csv, images_dir, CLASS_TO_IDX, train=False, channel_mode=mode)
        
        sampler = WeightedRandomSampler(train_dataset.get_sample_weights(), len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model = ResNetMIL(NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 与 StructureAwareMIL_v12 对齐的学习率策略（多 param-group）
        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': 2e-5},
            {'params': model.proj.parameters(), 'lr': 2e-4},
            {'params': model.transformer.parameters(), 'lr': 2e-4},
            {'params': model.attention.parameters(), 'lr': 2e-4},
            {'params': model.classifier.parameters(), 'lr': 2e-4},
            {'params': [model.pos_embed], 'lr': 2e-4}
        ], weight_decay=1e-3)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[2e-5, 2e-4, 2e-4, 2e-4, 2e-4, 2e-4],
            total_steps=len(train_loader) * MAX_EPOCHS,
            pct_start=0.1
        )

        best_val_f1 = 0
        no_improve_epochs = 0
        model_name = f"best_channel_ablation_v12_{mode}_seed{SEED_FOR_ABLATION}.pth"
        model_path = save_dir / model_name

        best_test_acc_at_val = 0.0
        best_test_f1_at_val = 0.0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
            test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, device)

            print(
                f"Epoch {epoch:3d} [{mode}] | "
                f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_test_acc_at_val = test_acc
                best_test_f1_at_val = test_f1
                torch.save(model.state_dict(), model_path)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch} [{mode}] (patience={PATIENCE})")
                break
                
        # 使用最优 val_f1 对应的保存权重，输出最终 test 指标
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        _, final_test_acc, final_test_f1, _, _ = evaluate(model, test_loader, criterion, device)

        results_dict[mode] = {
            'best_val_f1': best_val_f1,
            'best_test_acc': final_test_acc,
            'best_test_f1': final_test_f1
        }

    print("\n" + "="*60 + "\n🔥 CHANNEL ABLATION RESULTS 🔥\n" + "="*60)
    print(f"Baseline (Pure Grayscale)  [R]   -> Val F1: {results_dict['R']['best_val_f1']:.4f} | Test Acc: {results_dict['R']['best_test_acc']:.4f} | Test F1: {results_dict['R']['best_test_f1']:.4f}")
    print(f"Byte + Structure           [RG]  -> Val F1: {results_dict['RG']['best_val_f1']:.4f} | Test Acc: {results_dict['RG']['best_test_acc']:.4f} | Test F1: {results_dict['RG']['best_test_f1']:.4f}")
    print(f"Byte + Entropy             [RB]  -> Val F1: {results_dict['RB']['best_val_f1']:.4f} | Test Acc: {results_dict['RB']['best_test_acc']:.4f} | Test F1: {results_dict['RB']['best_test_f1']:.4f}")
    print(f"Proposed Method            [RGB] -> Val F1: {results_dict['RGB']['best_val_f1']:.4f} | Test Acc: {results_dict['RGB']['best_test_acc']:.4f} | Test F1: {results_dict['RGB']['best_test_f1']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
