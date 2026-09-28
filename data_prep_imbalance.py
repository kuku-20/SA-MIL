#!/usr/bin/env python3
"""重建极端不平衡数据划分"""
import os, csv, random
from pathlib import Path

PROJ = Path(__file__).resolve().parent
random.seed(42)

samples = []
# 恶意
for fd in (PROJ/'data'/'malware_images'/'images').iterdir():
    if fd.is_dir():
        for f in fd.glob('*.png'):
            samples.append((f.stem, 1, str(f.relative_to(PROJ))))
# 良性（原版成像路径）
for f in (PROJ/'data'/'benign_original'/'benign').glob('*.png'):
    samples.append((f.stem, 0, str(f.relative_to(PROJ))))

ben = [(s,i,p) for s,i,p in samples if i==0]
mal = [(s,i,p) for s,i,p in samples if i==1]
random.shuffle(ben); random.shuffle(mal)

def make_split(name, ben_limit, mal_limit):
    b = random.sample(ben, min(len(ben), ben_limit))
    m = random.sample(mal, min(len(mal), mal_limit))
    random.shuffle(b); random.shuffle(m)
    out = PROJ/'data'/'binary_experiment'/name
    out.mkdir(parents=True, exist_ok=True)
    
    for sn, vr, tr in [('train',0.7,0),('val',0.15,0.15),('test',0,0.15)]:
        def split_items(items):
            n = len(items)
            nv = max(1, int(n*0.15))
            nt = max(1, int(n*0.15))
            if sn=='train': return items[:n-nv-nt]
            elif sn=='val': return items[n-nv-nt:n-nt]
            else: return items[n-nt:]
        data = split_items(b) + split_items(m)
        random.shuffle(data)
        with open(out/f'{sn}.csv','w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['sha256','label','image_path'])
            for s,l,p in data: w.writerow([s,l,p])
        bc = sum(1 for _,l,_ in data if l==0)
        mc = sum(1 for _,l,_ in data if l==1)
        print(f'  {name}/{sn}: {len(data):5d} = 良性{bc:4d} + 恶意{mc:4d}')

print("场景: 极端不平衡 (比例 ~20:1)")
make_split('extreme_imbalance', ben_limit=int(len(ben)/3), mal_limit=len(mal))
