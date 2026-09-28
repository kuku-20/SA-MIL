import os
import sys
import hashlib
import shutil
import csv
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"

# 输入源
sources = [
    DATA / "benign_sources" / "extracted" / "Benign-PE-Dataset-main" / "B",
    DATA / "benign_sources" / "extracted" / "Benign-PE-Dataset-main" / "Dike",
    DATA / "benign_sources" / "limited_dike" / "DikeDataset-main" / "files" / "benign",
]

# 输出目录
out_dir = DATA / "benign_samples_pe" / "benign"
out_dir.mkdir(parents=True, exist_ok=True)

seen = set()
copied = 0
skipped = 0
errors = 0

for src_dir in sources:
    if not src_dir.exists():
        print(f"⚠️  源目录不存在: {src_dir}")
        continue
    
    for fpath in sorted(src_dir.iterdir()):
        if fpath.suffix.lower() not in ('.exe', '.dll'):
            continue
        if not fpath.is_file():
            continue
        
        try:
            # 计算 SHA256
            sha = hashlib.sha256()
            with open(fpath, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha.update(chunk)
            sha_hex = sha.hexdigest()
            
            if sha_hex in seen:
                skipped += 1
                continue
            
            seen.add(sha_hex)
            dst = out_dir / f"{sha_hex}.bin"
            shutil.copy2(fpath, dst)
            copied += 1
            
        except Exception as e:
            errors += 1
            print(f"❌  处理失败: {fpath.name} -> {e}")

print(f"\n{'='*50}")
print(f"合并完成!")
print(f"  复制: {copied} 个文件")
print(f"  去重跳过: {skipped} 个")
print(f"  错误: {errors} 个")
print(f"  输出目录: {out_dir}")
print(f"{'='*50}")

# 生成标签 CSV
csv_path = DATA / "benign_samples_pe" / "labels.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sha256', 'family'])
    for sha in sorted(seen):
        writer.writerow([sha, 'benign'])
print(f"  标签 CSV: {csv_path}")
print(f"  条目数: {len(seen)}")
