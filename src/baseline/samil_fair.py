"""
Fair-condition SA-MIL runner for Table 2 comparison.
=====================================================
在与 4 条同类基线（IMCFN / ViT / AB-MIL / TransMIL）完全相同的训练预算下评估 SA-MIL：

  - MAX_EPOCHS = 5
  - SEED       = 42 （单 seed）
  - 单模型，无 ensemble，无 TTA
  - 其余（数据划分、采样、损失、优化器、调度器）与 SA-MIL v12 完全一致

输出会追加写入 baseline/peer_results.json，key = "samil_fair"。
"""

import os
import sys
import csv
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm

# 复用 v12 的 Dataset / Model / collate_fn
# Path to project root (auto-detected)
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
from src.models import mil_v12 as v12  # type: ignore

RESULTS_JSON = _PROJ_ROOT / "results" / "tables" / "peer_results.json"
MAX_EPOCHS   = 15
SEED         = 42
BATCH_SIZE   = 8


def main():
    # 与 baseline_peer_methods 保持完全一致的类别发现流程
    images_dir = str(_PROJ_ROOT / "data" / "malware_images")
    train_csv  = f"{images_dir}/train.csv"
    val_csv    = f"{images_dir}/val.csv"
    test_csv   = f"{images_dir}/test.csv"

    families = set()
    for p in [train_csv, val_csv, test_csv]:
        with open(p, 'r') as f:
            for row in csv.DictReader(f):
                families.add(row['family'])
    families = sorted(families)

    # 注入 v12 的全局配置
    v12.CLASS_TO_IDX = {fam: i for i, fam in enumerate(families)}
    v12.IDX_TO_CLASS = {i: fam for fam, i in v12.CLASS_TO_IDX.items()}
    v12.NUM_CLASSES  = len(families)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Classes({v12.NUM_CLASSES}): {families}")

    v12.set_seed(SEED)

    train_ds = v12.MalwareDataset(train_csv, images_dir, v12.CLASS_TO_IDX, train=True)
    val_ds   = v12.MalwareDataset(val_csv,   images_dir, v12.CLASS_TO_IDX, train=False)
    test_ds  = v12.MalwareDataset(test_csv,  images_dir, v12.CLASS_TO_IDX, train=False)

    sampler = WeightedRandomSampler(train_ds.get_sample_weights(), len(train_ds), replacement=True)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=v12.collate_fn, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=v12.collate_fn, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=v12.collate_fn, num_workers=0)

    model = v12.ResNetMIL(v12.NUM_CLASSES).to(device)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = optim.AdamW([
        {'params': model.features.parameters(),    'lr': 2e-5},
        {'params': model.proj.parameters(),        'lr': 2e-4},
        {'params': model.transformer.parameters(), 'lr': 2e-4},
        {'params': model.attention.parameters(),   'lr': 2e-4},
        {'params': model.classifier.parameters(),  'lr': 2e-4},
        {'params': [model.pos_embed],              'lr': 2e-4}
    ], weight_decay=1e-3)
    sch = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[2e-5, 2e-4, 2e-4, 2e-4, 2e-4, 2e-4],
        total_steps=len(train_ld) * MAX_EPOCHS, pct_start=0.1
    )

    print(f"\n==== Training SA-MIL (FAIR: {MAX_EPOCHS} ep, seed={SEED}, no TTA/ensemble) ====")
    best_val_f1 = 0.0
    best_state  = None
    for ep in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = v12.train_epoch(model, train_ld, opt, crit, device, sch)
        va_loss, va_acc, va_f1, _, _ = v12.evaluate(model, val_ld, crit, device)
        print(f"  ep{ep:02d} | train_f1={tr_f1:.4f} acc={tr_acc:.4f} | val_f1={va_f1:.4f} acc={va_acc:.4f}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nLoading best-val model (val_f1={best_val_f1:.4f}) for test evaluation...")
    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, preds, labs = v12.evaluate(model, test_ld, crit, device)

    report = classification_report(
        labs, preds,
        target_names=[v12.IDX_TO_CLASS[i] for i in range(v12.NUM_CLASSES)],
        digits=4, zero_division=0
    )
    print(f"\n[RESULT] SA-MIL (FAIR)  test_acc={te_acc:.4f}  test_macro_f1={te_f1:.4f}")
    print(report)

    # 写入 peer_results.json
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    allres = {}
    if RESULTS_JSON.exists():
        try: allres = json.loads(RESULTS_JSON.read_text())
        except Exception: allres = {}
    allres["samil_fair"] = {
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "best_val_macro_f1": best_val_f1,
        "condition": f"{MAX_EPOCHS} epoch, seed={SEED}, single model, no TTA, no ensemble",
        "report": report,
    }
    RESULTS_JSON.write_text(json.dumps(allres, indent=2, ensure_ascii=False))
    print(f"[SAVED] -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
