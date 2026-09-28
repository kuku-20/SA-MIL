#!/usr/bin/env python3
"""Experiment 3: FPR analysis on obfuscated benign files.

Mirrors exp2_final.py (UPX-packed FPR analysis) but uses the obfuscated
samples in data/obfuscated_all/ and the balanced binary model.
"""
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.b2image import generate_pe_image_and_offset_map


PROJ = Path(__file__).resolve().parent
SRC_DIR = PROJ / 'data' / 'obfuscated_all'
IMG_DIR = PROJ / 'data' / 'exp3_images' / 'benign_obfuscated'
INFER_CSV = PROJ / 'data' / 'exp3_infer.csv'
PROB_CSV = PROJ / 'data' / 'exp3_probabilities.csv'
RESULT_JSON = PROJ / 'results' / 'obfuscation_results.json'
BALANCED_DIR = PROJ / 'data' / 'binary_experiment' / 'balanced'


def _worker(args):
    in_path, out_png, out_map = args
    try:
        generate_pe_image_and_offset_map(in_path, out_png, out_map, silent=True)
        if os.path.exists(out_map):
            os.remove(out_map)
        return 1 if os.path.exists(out_png) else 0
    except Exception:
        return 0


def _valid_png(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def step1_imaging():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for bin_file in sorted(SRC_DIR.glob('*.bin')):
        out_png = IMG_DIR / f'{bin_file.stem}.png'
        out_map = IMG_DIR / f'{bin_file.stem}_map.txt'
        if out_png.exists() and _valid_png(out_png):
            continue
        tasks.append((str(bin_file), str(out_png), str(out_map)))

    print(f'[1/2] Obfuscated imaging: {len(tasks)} files to generate')
    if tasks:
        ok = 0
        with Pool(4) as pool:
            for result in tqdm(pool.imap_unordered(_worker, tasks, chunksize=4),
                               total=len(tasks), desc='Imaging', unit='file', ncols=100):
                ok += result
        print(f'      generated: {ok}, failed: {len(tasks) - ok}')

    png_count = len(list(IMG_DIR.glob('*.png')))
    print(f'      total images: {png_count}')
    return png_count


def step2_inference():
    from models.mil_binary import BinaryDataset, ResNetMIL, collate_fn
    from torch.utils.data import DataLoader

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(8)

    model = ResNetMIL().to(device)
    ckpt = PROJ / 'results' / 'checkpoints' / 'best_mil_binary.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with open(INFER_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sha256', 'label', 'image_path'])
        for png in sorted(IMG_DIR.glob('*.png')):
            writer.writerow([png.stem, 0, str(png.relative_to(PROJ))])

    ds = BinaryDataset(str(INFER_CSV), train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f'[2/2] Inference on {len(ds)} obfuscated samples (device={device})')

    hashes, probs = [], []
    with torch.inference_mode():
        for windows, labels, masks in tqdm(loader):
            windows = windows.to(device)
            masks = masks.to(device)
            logits = model(windows, masks)
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())

    # hashes come from the dataset in CSV order; reconstruct from loaded samples
    hashes = [Path(p).stem for p, _ in ds.samples]
    probs = np.asarray(probs[:len(hashes)])
    assert len(hashes) == len(probs), 'hash/prob count mismatch'

    with open(PROB_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sha256', 'malware_prob'])
        for h, p in zip(hashes, probs):
            writer.writerow([h, f'{p:.6f}'])

    total = len(probs)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    fpr_rows = {}
    for th in thresholds:
        fp = int((probs >= th).sum())
        fpr_rows[str(th)] = {'fp': fp, 'fpr': fp / total * 100.0 if total else 0.0}

    train_benign, val_benign, test_benign = set(), set(), set()
    for split, target in [('train.csv', train_benign), ('val.csv', val_benign), ('test.csv', test_benign)]:
        with open(BALANCED_DIR / split) as f:
            for row in csv.DictReader(f):
                if int(row['label']) == 0:
                    target.add(row['sha256'])

    hash_set = set(hashes)
    overlap = {
        'train': len(hash_set & train_benign),
        'val': len(hash_set & val_benign),
        'test': len(hash_set & test_benign),
    }

    results = {
        'total': int(total),
        'imaged': len(hashes),
        'fpr_by_threshold': fpr_rows,
        'mean_malware_prob': float(probs.mean()),
        'std_malware_prob': float(probs.std()),
        'max_malware_prob': float(probs.max()),
        'balanced_split_overlap': overlap,
    }
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print('=' * 60)
    print('Experiment 3: obfuscated benign FPR analysis')
    print('=' * 60)
    print(f'  total samples: {total}')
    print(f'  mean malware prob: {results["mean_malware_prob"]:.4f}')
    print(f'  std: {results["std_malware_prob"]:.4f}')
    print('  threshold FPR:')
    for th in thresholds:
        row = fpr_rows[str(th)]
        print(f'    >= {th:.1f}: {row["fp"]}/{total} = {row["fpr"]:.2f}%')
    print(f'  balanced split overlap: {overlap}')
    print(f'  saved: {RESULT_JSON}')


def main():
    total_bins = len(list(SRC_DIR.glob('*.bin')))
    if total_bins == 0:
        print('Error: no obfuscated files found in data/obfuscated_all')
        return
    step1_imaging()
    step2_inference()


if __name__ == '__main__':
    main()
