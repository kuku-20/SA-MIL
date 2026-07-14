import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. 核心配置 (与你训练时保持完全一致)
# ==========================================


# Project root (auto-detected)

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

FIXED_WIDTH   = 256
WINDOW_H      = 224
WINDOW_STRIDE = 112
MAX_WINDOWS   = 12
HIDDEN_DIM    = 512

# 替换为你实际的路径
IMAGES_DIR = str(_PROJ_DIR / "data" / "malware_images")
_PROJ_DIR / "data" / "malware_images" / "test.csv""
_PROJ_DIR / "data" / "malware_images" / "train.csv" # 用于获取完整类别名"
_PROJ_DIR / "data" / "malware_images" / "val.csv""

# 你之前训练保存的最好的权重文件，选一个分数最高的
# 比如之前 v12 跑出来的 seed 123 或 channel ablation 里的 rgb 权重
MODEL_WEIGHT_PATH = str(_PROJ_DIR / "results" / "checkpoints" / "best_mil_v12_seed123.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================
# 2. 类别提取与数据集定义
# ==========================================
families = set()
for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f): families.add(row['family'])
families = sorted(list(families))
CLASS_TO_IDX = {fam: i for i, fam in enumerate(families)}
IDX_TO_CLASS = {i: fam for i, fam in enumerate(families)}
NUM_CLASSES = len(families)

class PureTestDataset(Dataset):
    def __init__(self, csv_path, images_dir, class_to_idx):
        self.images_dir = Path(images_dir)
        self.samples = []
        self.class_to_idx = class_to_idx

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sha256 = row['sha256']
                family = row['family']
                if family not in self.class_to_idx: continue

                if 'image_path' in row and row['image_path']:
                    img_path = self.images_dir / row['image_path']
                else:
                    img_path = self.images_dir / family / f"{sha256}.png"

                if img_path.exists():
                    self.samples.append((str(img_path), self.class_to_idx[family]))

        self.global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
        self.eval_transform = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self): return len(self.samples)

    def _extract_windows(self, img):
        w, h = img.size
        windows = [self.global_resize(img)] 
        img_resized = img
        new_h = h
        if w != FIXED_WIDTH:
            new_h = int(h * FIXED_WIDTH / w)
            img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)

        if new_h <= WINDOW_H:
            canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
            canvas.paste(img_resized, (0, 0))
            while len(windows) < MAX_WINDOWS:
                windows.append(canvas)
            return windows

        local_max_windows = MAX_WINDOWS - 1 
        max_cover_h = WINDOW_STRIDE * (local_max_windows - 1) + WINDOW_H

        if new_h <= max_cover_h:
            for i in range(local_max_windows):
                top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
                windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            return windows

        # 熵驱动采样
        img_np = np.array(img_resized)  
        row_scores = np.sum(img_np[:, :, 1:3], axis=(1, 2)) 
        search_stride = 32
        candidate_windows = []
        for top in range(0, new_h - WINDOW_H + 1, search_stride):
            window_score = np.sum(row_scores[top : top + WINDOW_H])
            candidate_windows.append((window_score, top))
            
        candidate_windows.sort(key=lambda x: x[0], reverse=True)
        selected_tops = [0]
        min_distance = WINDOW_H // 2  
        
        for score, top in candidate_windows:
            if len(selected_tops) >= local_max_windows: break
            if all(abs(top - sel_top) >= min_distance for sel_top in selected_tops):
                selected_tops.append(top)
                
        while len(selected_tops) < local_max_windows:
            fallback_top = max(0, new_h - WINDOW_H - (local_max_windows - len(selected_tops)) * search_stride)
            selected_tops.append(fallback_top)

        selected_tops.sort()
        for top in selected_tops[:local_max_windows]:
            windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))

        return windows

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        windows = self._extract_windows(img)
        # 这里只有 ToTensor 和 Normalize，无任何增强
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

# ==========================================
# 3. 模型定义 (保持一致)
# ==========================================
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
        backbone = models.resnet50()
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.avgpool = backbone.avgpool
        self.proj = nn.Sequential(nn.Linear(2048, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU(), nn.Dropout(0.2))
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

# ==========================================
# 4. 执行纯净测试 (No TTA)
# ==========================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading Model Weights from: {MODEL_WEIGHT_PATH}")
    
    if not os.path.exists(MODEL_WEIGHT_PATH):
        print(f"❌ 找不到权重文件 {MODEL_WEIGHT_PATH}，请修改脚本第28行的文件名为你真实存在的文件！")
        return

    test_dataset = PureTestDataset(TEST_CSV, IMAGES_DIR, CLASS_TO_IDX)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model = ResNetMIL(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE), strict=False)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for windows, labels, masks in tqdm(test_loader, desc="Pure Eval (No TTA)"):
            windows, labels, masks = windows.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
            logits = model(windows, masks)
            # 直接取 argmax，不包含任何 torch.flip 融合
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("\n============================================================")
    print("🔥 PURE TEST RESULTS (SINGLE MODEL, NO TTA) 🔥")
    print("============================================================")
    print(f"Pure Test Accuracy  : {acc:.4f}")
    print(f"Pure Test Macro F1  : {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]))

if __name__ == "__main__":
    main()
