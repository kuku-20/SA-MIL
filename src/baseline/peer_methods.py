

# Project root (auto-detected)

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

"""
Peer-method baselines for SA-MIL comparison (Table 2).
==========================================================
实现 4 条同类方法基线，与 SA-MIL 在完全相同的数据划分下对比：

  图像级范式（单图 224x224 分类，代表"全局缩放"路线）:
    - imcfn   : VGG16 (ImageNet-pretrained) + 3-layer MLP head (Vasan 2020 风格)
    - vit     : ViT-B/16 (ImageNet-pretrained) fine-tune

  MIL 范式（与 SA-MIL 完全相同的 12 实例采样包，仅聚合器不同）:
    - abmil   : ResNet-50 instance encoder + Gated Attention Pool (Ilse 2018)
                —— 相当于 SA-MIL 去掉 Transformer 缝合与位置编码
    - transmil: ResNet-50 instance encoder + CLS token + TransformerEncoder(2 layers)
                —— TransMIL 的简化实现（CLS 聚合而非门控注意力）

所有方法共享：
  - 相同 train/val/test CSV 划分
  - WeightedRandomSampler（长尾平衡）
  - CrossEntropyLoss + label_smoothing=0.1
  - AdamW + OneCycleLR
  - 30 epoch + EarlyStop(patience=10)
  - 按 val Macro-F1 选最佳 epoch，测试集最终评估

运行方式（逐个跑）：
  python baseline_peer_methods.py --method imcfn
  python baseline_peer_methods.py --method vit
  python baseline_peer_methods.py --method abmil
  python baseline_peer_methods.py --method transmil

或批量：
  python baseline_peer_methods.py --method all

结果会写入  baseline/peer_results.json  并打印 classification_report。
"""

import os
import sys
import csv
import json
import math
import random
import argparse
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
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Paths & Constants
# -----------------------------------------------------------------------------
_PROJ_DIR / "")"
IMAGES_DIR   = ROOT / "data" / "malware_images"
TRAIN_CSV    = IMAGES_DIR / "train.csv"
VAL_CSV      = IMAGES_DIR / "val.csv"
TEST_CSV     = IMAGES_DIR / "test.csv"
RESULTS_JSON = ROOT / "baseline" / "peer_results.json"

FIXED_WIDTH   = 256
WINDOW_H      = 224
WINDOW_STRIDE = 112
MAX_WINDOWS   = 12

HIDDEN_DIM  = 512
MAX_EPOCHS  = 15
PATIENCE    = 5
BATCH_SIZE  = 8   # bag 批大小；图像级方法会自动放大 4x
SEED        = 42

CLASS_TO_IDX = {}
IDX_TO_CLASS = {}
NUM_CLASSES  = 0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def discover_classes():
    global CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
    families = set()
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        with open(p, 'r') as f:
            for row in csv.DictReader(f):
                families.add(row['family'])
    families = sorted(families)
    CLASS_TO_IDX = {fam: i for i, fam in enumerate(families)}
    IDX_TO_CLASS = {i: fam for fam, i in CLASS_TO_IDX.items()}
    NUM_CLASSES  = len(families)
    return families


# =============================================================================
# Dataset 1: Image-level (IMCFN / ViT) —— 全局缩放至 224x224
# =============================================================================
class ImageLevelDataset(Dataset):
    def __init__(self, csv_path, train=True):
        self.samples = []
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                fam = row['family']
                if fam not in CLASS_TO_IDX: continue
                sha = row['sha256']
                rel = row.get('image_path') or f"{fam}/{sha}.png"
                p = IMAGES_DIR / rel
                if not p.exists():
                    p = IMAGES_DIR / fam / f"{sha}.png"
                if p.exists():
                    self.samples.append((str(p), CLASS_TO_IDX[fam]))
        self.class_counts = Counter([s[1] for s in self.samples])
        base = [transforms.Resize((224, 224))]
        if train:
            base += [transforms.RandomHorizontalFlip(0.5)]
        base += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.tf = transforms.Compose(base)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert('RGB')
        return self.tf(img), y

    def get_sample_weights(self):
        tot = sum(self.class_counts.values())
        return [tot / (NUM_CLASSES * self.class_counts[y]) for _, y in self.samples]


# =============================================================================
# Dataset 2: Bag-level (AB-MIL / TransMIL) —— 复用 SA-MIL 的显著性采样窗口
# =============================================================================
class BagDataset(Dataset):
    """与 StructureAwareMIL_v12.MalwareDataset 完全一致的实例包采样（确保公平）。"""
    def __init__(self, csv_path, train=True):
        self.samples = []
        self.train = train
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                fam = row['family']
                if fam not in CLASS_TO_IDX: continue
                sha = row['sha256']
                rel = row.get('image_path') or f"{fam}/{sha}.png"
                p = IMAGES_DIR / rel
                if not p.exists():
                    p = IMAGES_DIR / fam / f"{sha}.png"
                if p.exists():
                    self.samples.append((str(p), CLASS_TO_IDX[fam]))
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
        tot = sum(self.class_counts.values())
        return [tot / (NUM_CLASSES * self.class_counts[y]) for _, y in self.samples]

    def _extract_windows(self, img):
        w, h = img.size
        windows = [self.global_resize(img)]
        img_r = img; new_h = h
        if w != FIXED_WIDTH:
            new_h = int(h * FIXED_WIDTH / w)
            img_r = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)
        if new_h <= WINDOW_H:
            canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
            canvas.paste(img_r, (0, 0))
            while len(windows) < MAX_WINDOWS:
                windows.append(canvas)
            return windows
        local_max = MAX_WINDOWS - 1
        max_cover = WINDOW_STRIDE * (local_max - 1) + WINDOW_H
        if new_h <= max_cover:
            for i in range(local_max):
                top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
                windows.append(img_r.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            return windows
        img_np = np.array(img_r)
        row_scores = np.sum(img_np[:, :, 1:3], axis=(1, 2))
        cand = []
        for top in range(0, new_h - WINDOW_H + 1, 32):
            cand.append((np.sum(row_scores[top:top + WINDOW_H]), top))
        cand.sort(key=lambda x: x[0], reverse=True)
        tops = [0]; min_d = WINDOW_H // 2
        for _, top in cand:
            if len(tops) >= local_max: break
            if all(abs(top - t) >= min_d for t in tops):
                tops.append(top)
        while len(tops) < local_max:
            fb = new_h - WINDOW_H - (local_max - len(tops)) * 32
            tops.append(max(0, fb))
        tops.sort()
        for top in tops[:local_max]:
            windows.append(img_r.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
        return windows

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert('RGB')
        wins = self._extract_windows(img)
        if self.train:
            ts = [self.normalize(self.augment(w)) for w in wins]
        else:
            ts = [self.normalize(self.eval_transform(w)) for w in wins]
        return torch.stack(ts), y, len(ts)


def bag_collate(batch):
    wl, labels, nw = zip(*batch)
    B = len(wl); mx = max(nw); _, c, h, w = wl[0].shape
    windows = torch.zeros(B, mx, c, h, w)
    masks = torch.zeros(B, mx, dtype=torch.bool)
    for i, (x, n) in enumerate(zip(wl, nw)):
        windows[i, :n] = x; masks[i, :n] = True
    return windows, torch.tensor(labels), masks


# =============================================================================
# Models
# =============================================================================
class IMCFN(nn.Module):
    """Vasan 2020 风格：VGG16 ImageNet 预训练 + 定制全连接头。"""
    def __init__(self, num_classes):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool  = vgg.avgpool
        # 冻结前 10 个卷积层（Vasan 2020 迁移学习策略）
        for i, p in enumerate(self.features.parameters()):
            if i < 18: p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, 512),          nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


class ViTClassifier(nn.Module):
    """ViT-B/16 ImageNet 预训练 + 线性头。"""
    def __init__(self, num_classes):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_f = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_f, num_classes)

    def forward(self, x): return self.vit(x)


class _InstanceEncoder(nn.Module):
    """共用的 ResNet-50 实例编码器（与 SA-MIL 完全一致的 backbone 策略）。"""
    def __init__(self, out_dim=HIDDEN_DIM):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feat = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                  bb.layer1, bb.layer2, bb.layer3, bb.layer4)
        self.avg = bb.avgpool
        for i, c in enumerate(self.feat.children()):
            if i < 6:
                for p in c.parameters(): p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim), nn.LayerNorm(out_dim),
            nn.GELU(), nn.Dropout(0.2)
        )

    def forward(self, windows):
        B, N, c, h, w = windows.shape
        x = self.feat(windows.view(B * N, c, h, w))
        x = self.avg(x).flatten(1)
        return self.proj(x).view(B, N, -1)


class ABMIL(nn.Module):
    """Ilse 2018 Gated Attention MIL —— 无 Transformer 缝合。"""
    def __init__(self, num_classes):
        super().__init__()
        self.enc = _InstanceEncoder(HIDDEN_DIM)
        D = 128
        self.V = nn.Linear(HIDDEN_DIM, D)
        self.U = nn.Linear(HIDDEN_DIM, D)
        self.w = nn.Linear(D, 1)
        self.cls = nn.Sequential(nn.Dropout(0.5), nn.Linear(HIDDEN_DIM, num_classes))

    def forward(self, windows, masks):
        h = self.enc(windows)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~masks, -1e9)
        a = F.softmax(a, dim=-1)
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.cls(z)


class TransMIL(nn.Module):
    """TransMIL 简化实现：CLS token + 2 层 TransformerEncoder + CLS 分类。"""
    def __init__(self, num_classes):
        super().__init__()
        self.enc = _InstanceEncoder(HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=8, dim_feedforward=HIDDEN_DIM * 2,
            dropout=0.1, batch_first=True
        )
        # enable_nested_tensor=False: MPS 尚未实现 nested-tensor 快速路径所需算子
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.cls = nn.Sequential(nn.Dropout(0.5), nn.Linear(HIDDEN_DIM, num_classes))

    def forward(self, windows, masks):
        h = self.enc(windows)  # [B, N, D]
        B = h.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, h], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=masks.device)
        full_mask = torch.cat([cls_mask, masks], dim=1)
        x = self.transformer(x, src_key_padding_mask=~full_mask)
        return self.cls(self.norm(x[:, 0]))


# =============================================================================
# Training loops
# =============================================================================
def _run_epoch_image(model, loader, crit, opt, sch, device, train):
    model.train(train)
    preds, labs = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc=("Train" if train else "Eval"), leave=False):
            x, y = x.to(device), y.to(device)
            if train: opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            preds.extend(out.argmax(1).cpu().numpy()); labs.extend(y.cpu().numpy())
    return accuracy_score(labs, preds), f1_score(labs, preds, average='macro'), preds, labs


def _run_epoch_bag(model, loader, crit, opt, sch, device, train):
    model.train(train)
    preds, labs = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for windows, y, masks in tqdm(loader, desc=("Train" if train else "Eval"), leave=False):
            windows, y, masks = windows.to(device), y.to(device), masks.to(device)
            if train: opt.zero_grad()
            out = model(windows, masks)
            loss = crit(out, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            preds.extend(out.argmax(1).cpu().numpy()); labs.extend(y.cpu().numpy())
    return accuracy_score(labs, preds), f1_score(labs, preds, average='macro'), preds, labs


def train_and_eval(method: str, device):
    set_seed(SEED)
    is_bag = method in ("abmil", "transmil")

    if is_bag:
        train_ds = BagDataset(TRAIN_CSV, train=True)
        val_ds   = BagDataset(VAL_CSV,   train=False)
        test_ds  = BagDataset(TEST_CSV,  train=False)
        collate  = bag_collate
        bs = BATCH_SIZE
        run_epoch = _run_epoch_bag
    else:
        train_ds = ImageLevelDataset(TRAIN_CSV, train=True)
        val_ds   = ImageLevelDataset(VAL_CSV,   train=False)
        test_ds  = ImageLevelDataset(TEST_CSV,  train=False)
        collate  = None
        bs = BATCH_SIZE * 4  # 单图训练可放大 batch
        run_epoch = _run_epoch_image

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(), len(train_ds), replacement=True)
    train_ld = DataLoader(train_ds, batch_size=bs, sampler=sampler, collate_fn=collate, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False,  collate_fn=collate, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=bs, shuffle=False,  collate_fn=collate, num_workers=0)

    if method == "imcfn":    model = IMCFN(NUM_CLASSES)
    elif method == "vit":    model = ViTClassifier(NUM_CLASSES)
    elif method == "abmil":  model = ABMIL(NUM_CLASSES)
    elif method == "transmil": model = TransMIL(NUM_CLASSES)
    else: raise ValueError(method)
    model = model.to(device)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    sch  = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-4, total_steps=len(train_ld) * MAX_EPOCHS, pct_start=0.1
    )

    best_f1 = 0.0; best_state = None; bad = 0
    print(f"\n==== Training {method.upper()} ====")
    for ep in range(1, MAX_EPOCHS + 1):
        tr_acc, tr_f1, _, _ = run_epoch(model, train_ld, crit, opt, sch, device, True)
        va_acc, va_f1, _, _ = run_epoch(model, val_ld,   crit, opt, sch, device, False)
        print(f"  ep{ep:02d} | train_f1={tr_f1:.4f} acc={tr_acc:.4f} | val_f1={va_f1:.4f} acc={va_acc:.4f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop @ep{ep}")
                break

    model.load_state_dict(best_state)
    te_acc, te_f1, preds, labs = run_epoch(model, test_ld, crit, opt, sch, device, False)
    report = classification_report(labs, preds,
                                   target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)],
                                   digits=4, zero_division=0)
    print(f"\n[RESULT] {method.upper()}  test_acc={te_acc:.4f}  test_macro_f1={te_f1:.4f}")
    print(report)
    return {"method": method, "test_acc": te_acc, "test_macro_f1": te_f1,
            "best_val_macro_f1": best_f1, "report": report}


def save_result(result):
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    allres = {}
    if RESULTS_JSON.exists():
        try: allres = json.loads(RESULTS_JSON.read_text())
        except Exception: allres = {}
    allres[result["method"]] = {k: v for k, v in result.items() if k != "method"}
    RESULTS_JSON.write_text(json.dumps(allres, indent=2, ensure_ascii=False))
    print(f"[SAVED] -> {RESULTS_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["imcfn", "vit", "abmil", "transmil", "all"])
    args = ap.parse_args()

    discover_classes()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Classes({NUM_CLASSES}): {list(CLASS_TO_IDX.keys())}")

    methods = ["imcfn", "vit", "abmil", "transmil"] if args.method == "all" else [args.method]
    for m in methods:
        res = train_and_eval(m, device)
        save_result(res)


if __name__ == "__main__":
    main()
