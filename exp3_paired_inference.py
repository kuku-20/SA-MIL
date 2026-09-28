#!/usr/bin/env python3
"""Paired FPR analysis: original benign files vs obfuscated versions.

Run this after exp3_obfuscation.py has produced data/exp3_probabilities.csv.
The script infers malware probabilities on the original images for the same
hashes, then compares paired predictions with the obfuscated set.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.mil_binary import BinaryDataset, ResNetMIL, collate_fn
from torch.utils.data import DataLoader


PROJ = Path(__file__).resolve().parent
OBF_BIN_DIR = PROJ / 'data' / 'obfuscated_all'
ORIG_IMG_DIR = PROJ / 'data' / 'benign_original' / 'benign'
ORIG_INFER_CSV = PROJ / 'data' / 'exp3_original_infer.csv'
ORIG_PROB_CSV = PROJ / 'data' / 'exp3_original_probabilities.csv'
OBF_PROB_CSV = PROJ / 'data' / 'exp3_probabilities.csv'
PAIRED_CSV = PROJ / 'data' / 'exp3_paired_probabilities.csv'
RESULT_JSON = PROJ / 'results' / 'obfuscation_paired_results.json'
BALANCED_DIR = PROJ / 'data' / 'binary_experiment' / 'balanced'
THRESHOLDS = [0.5, 0.9]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_prob_csv(path):
    probs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            probs[row['sha256']] = float(row['malware_prob'])
    return probs


def _load_benign_split_hashes():
    splits = {s: set() for s in ['train', 'val', 'test']}
    for s in splits:
        with open(BALANCED_DIR / f'{s}.csv') as f:
            for row in csv.DictReader(f):
                if int(row['label']) == 0:
                    splits[s].add(row['sha256'])
    return splits


def _mc_nemar_pvalue(orig_fp, obf_fp):
    try:
        from scipy.stats import binomtest
    except Exception:
        return None
    b10 = int(((orig_fp == 1) & (obf_fp == 0)).sum())
    b01 = int(((orig_fp == 0) & (obf_fp == 1)).sum())
    n = b10 + b01
    if n == 0:
        return None
    return float(binomtest(min(b01, b10), n=n, p=0.5, alternative='two-sided').pvalue)


def step1_inference():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(8)

    hashes = sorted(p.stem for p in OBF_BIN_DIR.glob('*.bin'))
    missing = [h for h in hashes if not (ORIG_IMG_DIR / f'{h}.png').exists()]
    if missing:
        print(f'Error: {len(missing)} original images missing; aborting')
        return None

    with open(ORIG_INFER_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sha256', 'label', 'image_path'])
        for h in hashes:
            img = ORIG_IMG_DIR / f'{h}.png'
            writer.writerow([h, 0, str(img.relative_to(PROJ))])

    model = ResNetMIL().to(device)
    ckpt = PROJ / 'results' / 'checkpoints' / 'best_mil_binary.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ds = BinaryDataset(str(ORIG_INFER_CSV), train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f'[1/2] Original inference on {len(ds)} samples (device={device})')

    probs = []
    with torch.inference_mode():
        for windows, labels, masks in tqdm(loader):
            windows = windows.to(device)
            masks = masks.to(device)
            logits = model(windows, masks)
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())

    inferred_hashes = [Path(p).stem for p, _ in ds.samples]
    probs = np.asarray(probs[:len(inferred_hashes)])
    assert len(inferred_hashes) == len(probs), 'hash/prob count mismatch'

    with open(ORIG_PROB_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sha256', 'malware_prob'])
        for h, p in zip(inferred_hashes, probs):
            writer.writerow([h, f'{p:.6f}'])
    print(f'      saved: {ORIG_PROB_CSV}')
    return dict(zip(inferred_hashes, probs))


def step2_analysis(orig_probs):
    obf_probs = _load_prob_csv(OBF_PROB_CSV)
    splits = _load_benign_split_hashes()

    unchanged = set()
    for p in OBF_BIN_DIR.glob('*.bin'):
        if _sha256(p) == p.stem:
            unchanged.add(p.stem)

    hashes = sorted(set(orig_probs) & set(obf_probs))
    union = splits['train'] | splits['val'] | splits['test']
    groups = {
        'all': hashes,
        'changed_only': [h for h in hashes if h not in unchanged],
        'unchanged_only': [h for h in hashes if h in unchanged],
        'train_seen': [h for h in hashes if h in splits['train']],
        'val_seen': [h for h in hashes if h in splits['val']],
        'test_seen': [h for h in hashes if h in splits['test']],
        'unseen': [h for h in hashes if h not in union],
        'changed_unseen': [h for h in hashes if h not in unchanged and h not in union],
        'changed_train': [h for h in hashes if h not in unchanged and h in splits['train']],
    }

    with open(PAIRED_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sha256', 'changed', 'split', 'original_prob', 'obfuscated_prob'])
        for h in hashes:
            split = 'train' if h in splits['train'] else 'val' if h in splits['val'] else 'test' if h in splits['test'] else 'unseen'
            writer.writerow([h, int(h not in unchanged), split,
                             f'{orig_probs[h]:.6f}', f'{obf_probs[h]:.6f}'])

    results = {}
    for name, ids in groups.items():
        if not ids:
            continue
        orig = np.asarray([orig_probs[h] for h in ids])
        obf = np.asarray([obf_probs[h] for h in ids])
        entry = {
            'n': len(ids),
            'mean_original_prob': float(orig.mean()),
            'mean_obfuscated_prob': float(obf.mean()),
            'mean_prob_delta': float(obf.mean() - orig.mean()),
            'thresholds': {},
        }
        for th in THRESHOLDS:
            orig_fp = orig >= th
            obf_fp = obf >= th
            entry['thresholds'][str(th)] = {
                'original_fp': int(orig_fp.sum()),
                'original_fpr': float(orig_fp.mean() * 100.0),
                'obfuscated_fp': int(obf_fp.sum()),
                'obfuscated_fpr': float(obf_fp.mean() * 100.0),
                'fpr_delta_pp': float((obf_fp.mean() - orig_fp.mean()) * 100.0),
                'mc_nemar_p': _mc_nemar_pvalue(orig_fp, obf_fp),
            }
        results[name] = entry

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print('=' * 104)
    print('Paired FPR analysis: original vs obfuscated benign files')
    print('=' * 104)
    header = (f"{'group':<16} {'n':>5} {'orig@0.5':>9} {'obf@0.5':>9} "
              f"{'delta':>7} {'orig@0.9':>9} {'obf@0.9':>9} {'delta':>7} "
              f"{'mean_orig':>9} {'mean_obf':>9}")
    print(header)
    for name, entry in results.items():
        t05 = entry['thresholds']['0.5']
        t09 = entry['thresholds']['0.9']
        print(f"{name:<16} {entry['n']:>5} "
              f"{t05['original_fpr']:>8.2f}% {t05['obfuscated_fpr']:>8.2f}% {t05['fpr_delta_pp']:>+6.2f} "
              f"{t09['original_fpr']:>8.2f}% {t09['obfuscated_fpr']:>8.2f}% {t09['fpr_delta_pp']:>+6.2f} "
              f"{entry['mean_original_prob']:>9.4f} {entry['mean_obfuscated_prob']:>9.4f}")
    print()
    print(f'saved: {PAIRED_CSV}')
    print(f'saved: {RESULT_JSON}')


def main():
    if not OBF_PROB_CSV.exists():
        print('Error: data/exp3_probabilities.csv not found; run exp3_obfuscation.py first')
        return
    orig_probs = step1_inference()
    if orig_probs is not None:
        step2_analysis(orig_probs)


if __name__ == '__main__':
    main()
