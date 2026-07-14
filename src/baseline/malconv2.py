"""
MalConv2 baseline for SA-MIL comparison (Table 2).
====================================================
MalConv2 (Raff et al., 2021) removes the fixed 64KB truncation of MalConv,
processing the FULL byte sequence via strided convolutions + global max pooling.

Key differences from baseline_Malconv.py:
  - No truncation: processes the entire file byte stream
  - Architecture: embedding(8-dim) + multi-layer strided gated conv + global max pool
  - Same train/val/test splits as all other baselines

Usage:
  python baseline_malconv2.py

Results are appended to baseline/peer_results.json with key "malconv2".
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Paths & Constants
# -----------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT = _PROJ_ROOT
IMAGES_DIR   = ROOT / "data" / "malware_images"
TRAIN_CSV    = IMAGES_DIR / "train.csv"
VAL_CSV      = IMAGES_DIR / "val.csv"
TEST_CSV     = IMAGES_DIR / "test.csv"
RESULTS_JSON = ROOT / "results" / "tables" / "peer_results.json"

# MalConv2: NO truncation. We set a large upper bound for batching.
# Files larger than this are rare and will be right-padded or trimmed.
MAX_SEQ_LEN  = None  # dynamic per-sample; collate pads to batch max
BATCH_SIZE   = 4     # smaller batch due to variable-length sequences
MAX_EPOCHS   = 15
PATIENCE     = 4
SEED         = 42
EMB_DIM      = 8
CONV_CHANNELS = 128
NUM_CONV_LAYERS = 2
KERNEL_SIZE  = 512
STRIDE       = 512

FAMILIES = []
CLASS_TO_IDX = {}
IDX_TO_CLASS = {}
NUM_CLASSES  = 0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def discover_classes():
    global FAMILIES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
    fams = set()
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        with open(p, 'r') as f:
            for row in csv.DictReader(f):
                fams.add(row['family'])
    FAMILIES = sorted(fams)
    CLASS_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}
    IDX_TO_CLASS = {i: fam for fam, i in CLASS_TO_IDX.items()}
    NUM_CLASSES = len(FAMILIES)


# =============================================================================
# Dataset: Full-length byte sequences (no truncation)
# =============================================================================
class FullByteDataset(Dataset):
    """Load malware images, flatten to 1D byte sequences WITHOUT truncation."""
    def __init__(self, csv_path, train=True):
        self.samples = []
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                fam = row['family']
                if fam not in CLASS_TO_IDX:
                    continue
                sha = row['sha256']
                rel = row.get('image_path') or f"{fam}/{sha}.png"
                p = IMAGES_DIR / rel
                if not p.exists():
                    p = IMAGES_DIR / fam / f"{sha}.png"
                if p.exists():
                    self.samples.append((str(p), CLASS_TO_IDX[fam]))
        self.class_counts = Counter([s[1] for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert('L')
        byte_seq = np.array(img, dtype=np.uint8).flatten()
        byte_seq = torch.tensor(byte_seq, dtype=torch.long)
        return byte_seq, y

    def get_sample_weights(self):
        tot = sum(self.class_counts.values())
        return [tot / (NUM_CLASSES * self.class_counts[y]) for _, y in self.samples]


def collate_variable_length(batch):
    """Pad variable-length byte sequences to batch max length."""
    seqs, labels = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    # Round up to multiple of STRIDE for clean convolution
    max_len = ((max_len + STRIDE - 1) // STRIDE) * STRIDE
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s
    return padded, torch.tensor(labels, dtype=torch.long)


# =============================================================================
# MalConv2 Model
# =============================================================================
class MalConv2(nn.Module):
    """
    MalConv2 (Raff et al., 2021): Gated convolutions over full-length
    byte sequences with global channel-wise max pooling.

    Architecture:
      - Byte embedding (256 -> emb_dim)
      - N layers of [Conv1d + Gated Conv1d] with stride (no pooling between layers)
      - Global max pooling over time
      - FC classifier
    """
    def __init__(self, num_classes, emb_dim=EMB_DIM, channels=CONV_CHANNELS,
                 kernel_size=KERNEL_SIZE, stride=STRIDE, num_layers=NUM_CONV_LAYERS):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim, padding_idx=0)

        self.conv_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        in_ch = emb_dim
        for i in range(num_layers):
            s = stride if i == 0 else 1
            k = kernel_size if i == 0 else 3
            self.conv_layers.append(
                nn.Conv1d(in_ch, channels, k, stride=s, padding=k // 2, bias=True)
            )
            self.gate_layers.append(
                nn.Conv1d(in_ch, channels, k, stride=s, padding=k // 2, bias=True)
            )
            in_ch = channels

        self.classifier = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, L] long tensor of byte values
        x = self.emb(x)          # [B, L, emb_dim]
        x = x.transpose(1, 2)    # [B, emb_dim, L]

        for conv, gate in zip(self.conv_layers, self.gate_layers):
            x = F.relu(conv(x)) * torch.sigmoid(gate(x))

        # Global max pooling over time dimension
        x = x.max(dim=2)[0]      # [B, channels]
        return self.classifier(x)


# =============================================================================
# Training
# =============================================================================
def run_epoch(model, loader, crit, opt, sch, device, train):
    model.train(train)
    preds_all, labs_all = [], []
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, labels in tqdm(loader, desc=("Train" if train else "Eval"), leave=False):
            seqs, labels = seqs.to(device), labels.to(device)
            if train:
                opt.zero_grad()
            out = model(seqs)
            loss = crit(out, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sch is not None:
                    sch.step()
            total_loss += loss.item() * labels.size(0)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labs_all.extend(labels.cpu().numpy())
    n = len(labs_all)
    acc = accuracy_score(labs_all, preds_all)
    f1 = f1_score(labs_all, preds_all, average='macro')
    return acc, f1, preds_all, labs_all


def main():
    set_seed(SEED)
    discover_classes()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Classes({NUM_CLASSES}): {FAMILIES}")

    train_ds = FullByteDataset(TRAIN_CSV, train=True)
    val_ds   = FullByteDataset(VAL_CSV, train=False)
    test_ds  = FullByteDataset(TEST_CSV, train=False)

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(), len(train_ds), replacement=True)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=collate_variable_length, num_workers=0)
    val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_variable_length, num_workers=0)
    test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_variable_length, num_workers=0)

    model = MalConv2(NUM_CLASSES).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=len(train_ld) * MAX_EPOCHS, pct_start=0.1
    )

    print(f"\n==== Training MalConv2 (full-length, no truncation) ====")
    print(f"     {MAX_EPOCHS} epochs, seed={SEED}, batch={BATCH_SIZE}")
    print(f"     EMB={EMB_DIM}, channels={CONV_CHANNELS}, kernel={KERNEL_SIZE}, stride={STRIDE}")

    best_f1 = 0.0
    best_state = None
    bad = 0

    for ep in range(1, MAX_EPOCHS + 1):
        tr_acc, tr_f1, _, _ = run_epoch(model, train_ld, crit, opt, sch, device, True)
        va_acc, va_f1, _, _ = run_epoch(model, val_ld, crit, None, None, device, False)
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

    print(f"\nLoading best-val model (val_f1={best_f1:.4f}) for test evaluation...")
    model.load_state_dict(best_state)
    te_acc, te_f1, preds, labs = run_epoch(model, test_ld, crit, None, None, device, False)

    report = classification_report(
        labs, preds,
        target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)],
        digits=4, zero_division=0
    )
    print(f"\n[RESULT] MalConv2  test_acc={te_acc:.4f}  test_macro_f1={te_f1:.4f}")
    print(report)

    # Save to peer_results.json
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    allres = {}
    if RESULTS_JSON.exists():
        try:
            allres = json.loads(RESULTS_JSON.read_text())
        except Exception:
            allres = {}
    allres["malconv2"] = {
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "best_val_macro_f1": best_f1,
        "condition": f"{MAX_EPOCHS} epoch max, patience={PATIENCE}, seed={SEED}, full-length (no truncation)",
        "report": report,
    }
    RESULTS_JSON.write_text(json.dumps(allres, indent=2, ensure_ascii=False))
    print(f"[SAVED] -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
