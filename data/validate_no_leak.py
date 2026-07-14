import pandas as pd
import os



# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent

csv_files = {
    _PROJ_DIR / "data" / "malware_images" / "train.csv',"
    _PROJ_DIR / "data" / "malware_images" / "val.csv',"
    _PROJ_DIR / "data" / "malware_images" / "test.csv',"
}

# 读取三个数据集
data = {}
for split, path in csv_files.items():
    data[split] = pd.read_csv(path)
    print(f"[{split.upper()}] 样本数: {len(data[split])}")

# 提取 SHA-256（第一列）
sha256_train = set(data['train']['sha256'].unique())
sha256_val = set(data['val']['sha256'].unique())
sha256_test = set(data['test']['sha256'].unique())

# 检查重叠
leak_train_val = sha256_train & sha256_val
leak_train_test = sha256_train & sha256_test
leak_val_test = sha256_val & sha256_test

print("\n📊 数据泄漏检查:")
print(f"  Train ∩ Val:  {len(leak_train_val)} 重复")
print(f"  Train ∩ Test: {len(leak_train_test)} 重复")
print(f"  Val ∩ Test:   {len(leak_val_test)} 重复")

if len(leak_train_val) == 0 and len(leak_train_test) == 0 and len(leak_val_test) == 0:
    print("\n✅ 通过！没有数据泄漏，满足去重要求。")
else:
    print("\n❌ 失败！存在数据泄漏，需要重新划分数据集。")
    if leak_train_val:
        print(f"   重复样本 (Train-Val): {list(leak_train_val)[:5]}")