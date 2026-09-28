#!/usr/bin/env python3
"""实验2: UPX加壳假阳性分析（原版b2image成像）"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ====== Step 1: 原版成像 ======
from multiprocessing import Pool
from preprocessing.b2image import generate_pe_image_and_offset_map

def worker(args):
    try:
        generate_pe_image_and_offset_map(args[0], args[1], args[2], silent=True)
        if os.path.exists(args[2]): os.remove(args[2])
        return 1
    except:
        return 0

def step1_imaging():
    base = os.path.dirname(__file__)
    tasks = []
    with open(f'{base}/data/exp2_packed/labels.csv') as f:
        for row in csv.DictReader(f):
            in_path = f'{base}/data/exp2_packed/{row["family"]}/{row["sha256"]}.bin'
            out_dir = f'{base}/data/exp2_images/{row["family"]}'
            os.makedirs(out_dir, exist_ok=True)
            out_png = f'{out_dir}/{row["sha256"]}.png'
            out_map = f'{out_dir}/{row["sha256"]}_map.txt'
            if os.path.exists(out_png): continue
            tasks.append((in_path, out_png, out_map))
    
    print(f'[1/2] 原版成像: {len(tasks)} 个文件')
    with Pool(4) as p:
        res = p.map(worker, tasks)
    print(f'      完成: 成功 {sum(res)}, 失败 {len(tasks)-sum(res)}')
    img_count = len(os.listdir(f'{base}/data/exp2_images/benign_packed'))
    print(f'      图像: {img_count}')
    return img_count

# ====== Step 2: 推理 ======
def step2_inference():
    from models.mil_binary import ResNetMIL, BinaryDataset, collate_fn
    from torch.utils.data import DataLoader
    import torch, numpy as np
    from pathlib import Path
    from tqdm import tqdm

    base = Path(__file__).resolve().parent
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'\n[2/2] 推理设备: {device}')

    model = ResNetMIL().to(device)
    ckpt = base / 'results' / 'checkpoints' / 'best_mil_binary.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 创建推理CSV
    infer_csv = str(base / 'data' / 'exp2_infer.csv')
    with open(infer_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sha256','label','image_path'])
        for png in sorted((base / 'data' / 'exp2_images' / 'benign_packed').glob('*.png')):
            w.writerow([png.stem, 0, str(png.relative_to(base))])

    ds = BinaryDataset(infer_csv, train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f'      推理样本: {len(ds)} 个')

    all_probs = []
    with torch.no_grad():
        for w, l, m in tqdm(loader):
            w, m = w.to(device), m.to(device)
            logits = model(w, m)
            probs = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
            all_probs.extend(probs)

    total = len(all_probs)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    fp = int(sum(preds))

    print(f'\n{"="*60}')
    print(f'实验2: UPX加壳假阳性分析（原版成像）')
    print(f'{"="*60}')
    print(f'  测试样本: {total} 个')
    print(f'  误判为恶意(FP): {fp} 个')
    print(f'  FPR: {fp/total*100:.2f}%')
    print(f'  平均置信度: {np.mean(all_probs):.4f}')
    print(f'\n  不同阈值 FPR:')
    for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p = sum(1 for x in all_probs if x >= th)
        print(f'    >= {th:.1f}: {p}/{total} = {p/total*100:.2f}%')

def main():
    # 检查加壳文件是否存在
    base = os.path.dirname(__file__)
    packed_dir = f'{base}/data/exp2_packed/benign_packed'
    packed_count = len([f for f in os.listdir(packed_dir) if f.endswith('.bin')]) if os.path.exists(packed_dir) else 0
    if packed_count == 0:
        print('错误: 未找到加壳文件，请先运行UPX加壳')
        return
    
    step1_imaging()
    step2_inference()

if __name__ == '__main__':
    main()
