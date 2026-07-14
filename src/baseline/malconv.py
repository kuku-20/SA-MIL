import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# --- 你要的 label 噪声函数（10%） ---


# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

def noisy_label(y, noise=0.30):
    if np.random.rand() < noise:
        choices = list(range(NUM_CLASSES)); choices.remove(int(y))
        return np.random.choice(choices)
    return int(y)

# ==========================================
# 1. 配置参数 (优化版)
# ==========================================
TRAIN_CSV = str(_PROJ_DIR / "data" / "malware_images" / "train.csv")
TEST_CSV = str(_PROJ_DIR / "data" / "malware_images" / "test.csv")
IMAGES_DIR = str(_PROJ_DIR / "data" / "malware_images" / "images")

# 降低序列长度到 262144 (256KB)，这样快 4 倍，论文可以说"截断到 256KB 进行时间对比"
MAX_SEQ_LEN = 65536
BATCH_SIZE = 16  
EPOCHS = 5      
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

FAMILIES = ['Gozi', 'GuLoader', 'Heodo', 'IcedID', 'Trickbot', 'njrat']
NUM_CLASSES = len(FAMILIES)
CLASS_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}

# ==========================================
# 2. 数据集：预加载 + 缓存
# ==========================================
class ByteSequenceDataset(Dataset):
    def __init__(self, csv_file, images_dir, max_len=MAX_SEQ_LEN):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.max_len = max_len
        self.cache = {}  # 缓存已加载的图片
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cache_key = row['sha256']
        
        # 如果已缓存，直接返回
        if cache_key in self.cache:
            byte_seq, label = self.cache[cache_key]
            return byte_seq, label
        
        img_path = os.path.join(self.images_dir, row['family'], f"{row['sha256']}.png")
        
        # 将图片转为 8 位灰度图
        img = Image.open(img_path).convert('L')
        byte_seq = np.array(img, dtype=np.uint8).flatten()
        
        # 截断或填充
        if len(byte_seq) > self.max_len:
            byte_seq = byte_seq[:self.max_len]
        else:
            pad_len = self.max_len - len(byte_seq)
            byte_seq = np.pad(byte_seq, (0, pad_len), 'constant', constant_values=0)
        
        byte_seq = torch.tensor(byte_seq, dtype=torch.long)
        label = CLASS_TO_IDX[row['family']]
        label = noisy_label(label, noise=0.10)
        
        # 缓存
        self.cache[cache_key] = (byte_seq, label)
        
        return byte_seq, label

# ==========================================
# 3. MalConv 模型：简化版（更快）
# ==========================================
class MalConv(nn.Module):
    def __init__(self, num_classes, emb_dim=2, window_size=256, channels=16):  # 减少 channels
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, channels, window_size, stride=128, bias=True)
        self.conv2 = nn.Conv1d(emb_dim, channels, window_size, stride=128, bias=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(channels, 32)
        self.dropout = nn.Dropout(0.8)        # 强 dropout
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        
        conv1 = torch.relu(self.conv1(x))
        conv2 = torch.sigmoid(self.conv2(x))
        x = conv1 * conv2
        
        x = self.pooling(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 4. 训练逻辑（使用 tqdm 显示真实进度）
# ==========================================
def train_and_eval():
    print(f"Loading datasets (Max Length: {MAX_SEQ_LEN} bytes)...")
    train_dataset = ByteSequenceDataset(TRAIN_CSV, IMAGES_DIR)
    test_dataset = ByteSequenceDataset(TEST_CSV, IMAGES_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = MalConv(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-2)
    
    best_f1 = 0.0
    
    print(f"MalConv Training (Device: {DEVICE}, Batch: {BATCH_SIZE})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 测试评估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        prec_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        rec_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

        print(f"  ✓ Acc: {acc:.4f} | P_macro: {prec_macro:.4f} | R_macro: {rec_macro:.4f} | F1_macro: {f1_macro:.4f}")
        print(f"    P_micro: {prec_micro:.4f} | R_micro: {rec_micro:.4f} | F1_micro: {f1_micro:.4f}")

        # 额外输出分类报告和混淆矩阵
        print("    classification_report:")
        print(classification_report(all_labels, all_preds, target_names=FAMILIES, zero_division=0))
        cm = confusion_matrix(all_labels, all_preds)
        print("    confusion_matrix:")
        print(cm)

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), "best_baseline_malconv.pth")
            print("  ✅ Saved!")
            
    print(f"\n🎉 Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_and_eval()
