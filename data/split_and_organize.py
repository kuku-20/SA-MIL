import os
import csv
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

def infer_family_from_path(path: Path, all_dir: Path) -> str:
    rel = path.relative_to(all_dir)
    # 默认取 all 下第一层目录作为 family；若没有子目录则归到 UNKNOWN
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return "UNKNOWN"

def sha_from_png_name(name: str) -> str:
    # 例如 abc.png -> abc
    return Path(name).stem

def map_candidates_for_sha(sha: str):
    # 兼容两种常见命名
    return [f"{sha}_map.txt", f"{sha}_offset_map.txt"]

def collect_pairs(all_dir: Path):
    png_files = list(all_dir.rglob("*.png"))
    txt_files = list(all_dir.rglob("*.txt"))

    # 建立 map 索引：key=(family, filename)
    map_index = defaultdict(list)
    for t in txt_files:
        fam = infer_family_from_path(t, all_dir)
        map_index[(fam, t.name)].append(t)

    pairs = []
    for p in png_files:
        fam = infer_family_from_path(p, all_dir)
        sha = sha_from_png_name(p.name)
        found_map = None

        for cand in map_candidates_for_sha(sha):
            lst = map_index.get((fam, cand), [])
            if lst:
                found_map = lst[0]
                break

        # 若同family找不到，尝试全局兜底
        if found_map is None:
            for cand in map_candidates_for_sha(sha):
                # 全局搜同名
                matches = [t for t in txt_files if t.name == cand]
                if matches:
                    found_map = matches[0]
                    break

        if found_map:
            pairs.append({
                "sha256": sha,
                "family": fam,
                "img_path": p,
                "map_path": found_map
            })

    return pairs

def split_family_items(items, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # 极小类保护
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_val == 0: n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def copy_pair(item, out_root: Path):
    fam = item["family"]
    sha = item["sha256"]

    img_dst_dir = out_root / "images" / fam
    map_dst_dir = out_root / "offset_maps" / fam
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    map_dst_dir.mkdir(parents=True, exist_ok=True)

    img_dst = img_dst_dir / f"{sha}.png"
    # 保持 map 原始命名
    map_dst = map_dst_dir / item["map_path"].name

    shutil.copy2(item["img_path"], img_dst)
    shutil.copy2(item["map_path"], map_dst)

    return img_dst, map_dst

def write_csv(rows, csv_path: Path, out_root: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "family", "image_path", "offset_map_path"])
        for r in rows:
            writer.writerow([
                r["sha256"],
                r["family"],
                str(Path(r["image_path"]).relative_to(out_root)),
                str(Path(r["offset_map_path"]).relative_to(out_root)),
            ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-dir", type=str, required=True, help="包含混合图片和map的目录")
    parser.add_argument("--out-root", type=str, required=True, help="输出根目录")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_dir = Path(args.all_dir).resolve()
    out_root = Path(args.out_root).resolve()

    print(f"[1/4] 扫描样本: {all_dir}")
    pairs = collect_pairs(all_dir)
    if not pairs:
        print("未找到可配对的样本（png + map）。")
        return
    print(f"配对成功: {len(pairs)}")

    print("[2/4] 拷贝并分离到 images / offset_maps")
    copied = []
    for item in pairs:
        img_dst, map_dst = copy_pair(item, out_root)
        copied.append({
            "sha256": item["sha256"],
            "family": item["family"],
            "image_path": img_dst,
            "offset_map_path": map_dst
        })

    print("[3/4] 按 family 分层划分 train/val/test")
    fam_dict = defaultdict(list)
    for x in copied:
        fam_dict[x["family"]].append(x)

    train_rows, val_rows, test_rows = [], [], []
    for fam, items in fam_dict.items():
        tr, va, te = split_family_items(
            items,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        train_rows.extend(tr)
        val_rows.extend(va)
        test_rows.extend(te)

    print("[4/4] 写出标签文件")
    write_csv(train_rows, out_root / "train.csv", out_root)
    write_csv(val_rows, out_root / "val.csv", out_root)
    write_csv(test_rows, out_root / "test.csv", out_root)

    print("完成。")
    print(f"train: {len(train_rows)}")
    print(f"val:   {len(val_rows)}")
    print(f"test:  {len(test_rows)}")
    print(f"total: {len(copied)}")
    print(f"输出目录: {out_root}")

if __name__ == "__main__":
    main()