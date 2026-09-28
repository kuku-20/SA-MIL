"""
SA-MIL 二分类模型（良性 vs 恶意）
基于 mil_v12.py 改编，支持 FPR/AUC 评估
"""
import os, sys, csv, math, random, json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

PROJ_DIR = Path(__file__).resolve().parent.parent.parent

FIXED_WIDTH   = 256
WINDOW_H      = 224
WINDOW_STRIDE = 112
MAX_WINDOWS   = 12
HIDDEN_DIM    = 512
MAX_EPOCHS    = 5
PATIENCE      = 5
BATCH_SIZE    = 16
SEED          = 42

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class BinaryDataset(Dataset):
    """二分类数据加载器: 读取 CSV 中的 image_path + label"""
    def __init__(self, csv_path, train=True):
        self.samples = []
        self.train = train
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                label = int(row['label'])
                img_path = row['image_path']
                full_path = PROJ_DIR / img_path
                if full_path.exists():
                    self.samples.append((str(full_path), label))
        self.labels = [s[1] for s in self.samples]
        self.class_counts = Counter(self.labels)
        self.global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
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
        return [total / (2 * self.class_counts[label]) for _, label in self.samples]

    def _extract_windows(self, img):
        w, h = img.size
        windows = [self.global_resize(img)]
        img_r, new_h = img, h
        if w != FIXED_WIDTH:
            new_h = int(h * FIXED_WIDTH / w)
            img_r = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)
        if new_h <= WINDOW_H:
            canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
            canvas.paste(img_r, (0, 0))
            while len(windows) < MAX_WINDOWS: windows.append(canvas)
            return windows
        local_max = MAX_WINDOWS - 1
        max_cover = WINDOW_STRIDE * (local_max - 1) + WINDOW_H
        if new_h <= max_cover:
            for i in range(local_max):
                top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
                windows.append(img_r.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            return windows
        # 超大文件: 熵驱动采样
        img_np = np.array(img_r)
        row_scores = np.sum(img_np[:, :, 1:3], axis=(1, 2))
        cand = []
        for top in range(0, new_h - WINDOW_H + 1, 32):
            cand.append((np.sum(row_scores[top:top+WINDOW_H]), top))
        cand.sort(key=lambda x: x[0], reverse=True)
        tops = [0]
        for _, top in cand:
            if len(tops) >= local_max: break
            if all(abs(top-t) >= WINDOW_H//2 for t in tops): tops.append(top)
        while len(tops) < local_max:
            fb = max(0, new_h - WINDOW_H - (local_max - len(tops)) * 32)
            tops.append(fb)
        tops.sort()
        for top in tops[:local_max]:
            windows.append(img_r.crop((0, top, FIXED_WIDTH, top+WINDOW_H)))
        return windows

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        windows = self._extract_windows(img)
        if self.train:
            ts = [self.normalize(self.augment(w)) for w in windows]
        else:
            ts = [self.normalize(self.eval_transform(w)) for w in windows]
        return torch.stack(ts), label, len(windows)

def collate_fn(batch):
    wl, labels, nw = zip(*batch)
    B = len(wl); mx = max(nw); _,c,h,w = wl[0].shape
    windows = torch.zeros(B, mx, c, h, w)
    masks = torch.zeros(B, mx, dtype=torch.bool)
    for i, (x, n) in enumerate(zip(wl, nw)):
        windows[i, :n] = x; masks[i, :n] = True
    return windows, torch.tensor(labels), masks

# ============== 模型（与 mil_v12 相同的 SA-MIL 架构） ==============
class SafeTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask):
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2); src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2); src = self.norm2(src)
        return src

class ResNetMIL(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                      bb.layer1, bb.layer2, bb.layer3, bb.layer4)
        self.avgpool = bb.avgpool
        for i, c in enumerate(self.features.children()):
            if i < 6:
                for p in c.parameters(): p.requires_grad = False
        self.proj = nn.Sequential(nn.Linear(2048, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM),
                                  nn.GELU(), nn.Dropout(0.2))
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_WINDOWS, HIDDEN_DIM) * 0.02)
        self.transformer = SafeTransformerLayer(HIDDEN_DIM, 4, HIDDEN_DIM*2, 0.1)
        self.attention = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(HIDDEN_DIM, 2))

    def forward(self, windows, masks):
        B, N, c, h, w = windows.shape
        x = self.features(windows.view(B*N, c, h, w))
        x = self.avgpool(x).flatten(1)
        x = self.proj(x).view(B, N, -1)
        x = x + self.pos_embed[:, :N, :]
        x = self.transformer(x, src_key_padding_mask=~masks)
        attn = self.attention(x).squeeze(-1).masked_fill(~masks, -1e4)
        attn_w = F.softmax(attn, dim=-1)
        pooled = (attn_w.unsqueeze(-1) * x).sum(dim=1)
        return self.classifier(pooled)

# ============== 训练与评估 ==============
@torch.no_grad()
def evaluate_binary(model, loader, device):
    """评估二分类模型，返回详细指标"""
    model.eval()
    all_probs, all_labels = [], []
    for windows, labels, masks in tqdm(loader, desc="Eval", leave=False):
        windows, labels, masks = windows.to(device), labels.to(device), masks.to(device)
        with torch.amp.autocast(device.type):
                logits = model(windows, masks)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    # AUC (需要两类都存在)
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        # FPR @ TPR=95%
        fpr_at_95tpr = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 1.0
    else:
        auc = 0.5
        fpr_at_95tpr = 1.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    tn = np.sum((all_labels == 0) & (preds == 0))
    fp = np.sum((all_labels == 0) & (preds == 1))
    fn = np.sum((all_labels == 1) & (preds == 0))
    tp = np.sum((all_labels == 1) & (preds == 1))
    fpr_val = fp / max(fp + tn, 1)
    fnr_val = fn / max(fn + tp, 1)

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'auc': auc, 'fpr': fpr_val, 'fnr': fnr_val,
        'fpr_at_95tpr': fpr_at_95tpr,
        'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }, all_labels, all_probs

def train_epoch(model, loader, optimizer, criterion, device, scheduler, scaler=None):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for windows, labels, masks in tqdm(loader, desc="Train", leave=False):
        windows, labels, masks = windows.to(device), labels.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device.type):
                logits = model(windows, masks)
        loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')
    if device.type == "mps":
        torch.mps.empty_cache()
    return total_loss / len(loader), acc, f1

@torch.no_grad()
def evaluate_ensemble_tta(model_paths, loader, device):
    models = []
    for path in model_paths:
        m = ResNetMIL().to(device)
        m.load_state_dict(torch.load(path))
        m.eval(); models.append(m)
    all_probs, all_labels = [], []
    for windows, labels, masks in tqdm(loader, desc="Ensemble+TTA"):
        windows, labels, masks = windows.to(device), labels.to(device), masks.to(device)
        batch_probs = torch.zeros(windows.size(0)).to(device)
        for m in models:
            logits1 = m(windows, masks)
            windows_flip = torch.flip(windows, dims=[-1])
            logits2 = m(windows_flip, masks)
            probs = (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1))[:, 1] / 2.0
            batch_probs += probs
        batch_probs /= len(models)
        all_probs.extend(batch_probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_probs = np.array(all_probs); all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0, average='binary')
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return acc, f1, auc, all_preds, all_labels

# ============== Main ==============
def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base = PROJ_DIR / "data" / "binary_experiment" / "balanced"
    train_csv = base / "train.csv"
    val_csv = base / "val.csv"
    test_csv = base / "test.csv"
    model_save_dir = PROJ_DIR / "results" / "checkpoints"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BinaryDataset(train_csv, train=True)
    val_ds = BinaryDataset(val_csv, train=False)
    test_ds = BinaryDataset(test_csv, train=False)

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(),
                                    len(train_ds), replacement=True)
    train_ld = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ResNetMIL().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 2e-5},
        {'params': model.proj.parameters(), 'lr': 2e-4},
        {'params': model.transformer.parameters(), 'lr': 2e-4},
        {'params': model.attention.parameters(), 'lr': 2e-4},
        {'params': model.classifier.parameters(), 'lr': 2e-4},
        {'params': [model.pos_embed], 'lr': 2e-4}
    ], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=[2e-5,2e-4,2e-4,2e-4,2e-4,2e-4],
        total_steps=len(train_ld)*MAX_EPOCHS, pct_start=0.1)

    scaler = torch.amp.GradScaler(device.type)
    best_auc = 0
    patience_counter = 0
    model_path = model_save_dir / "best_mil_binary.pth"

    print(f"\n训练开始 ({len(train_ds)} train, {len(val_ds)} val)")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train F1':>8} | {'Val AUC':>7} | {'Val F1':>7} | {'Val FPR':>7}")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_ld, optimizer, criterion, device, scheduler, scaler)
        val_metrics, _, _ = evaluate_binary(model, val_ld, device)

        print(f"{epoch:6d} | {train_loss:10.4f} | {train_f1:8.4f} | {val_metrics['auc']:7.4f} | {val_metrics['f1']:7.4f} | {val_metrics['fpr']:7.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # 最终测试
    model.load_state_dict(torch.load(model_path))
    test_metrics, test_labels, test_probs = evaluate_binary(model, test_ld, device)

    print(f"\n{'='*60}")
    print(f"  测试集结果 (balanced)")
    print(f"{'='*60}")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1:     {test_metrics['f1']:.4f}")
    print(f"  AUC:    {test_metrics['auc']:.4f}")
    print(f"  FPR:    {test_metrics['fpr']:.4f}")
    print(f"  FNR:    {test_metrics['fnr']:.4f}")
    print(f"  FPR@TPR95: {test_metrics['fpr_at_95tpr']:.4f}")
    cm = test_metrics['confusion']
    print(f"\n  混淆矩阵:")
    print(f"           预测良性   预测恶意")
    print(f"  实际良性  {cm['tn']:>5}       {cm['fp']:>5}")
    print(f"  实际恶意  {cm['fn']:>5}       {cm['tp']:>5}")

    # 保存结果
    results_path = PROJ_DIR / "results" / "binary_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  结果已保存: {results_path}")

if __name__ == "__main__":
    main()
