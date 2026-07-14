import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)  # 避免无 GUI 环境下 Matplotlib 失败
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# seaborn 只用于热力图美化；如果环境没有 seaborn，则用 matplotlib 退化实现
try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:
    sns = None

# 新数据与模型路径配置
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "malware_images")
CLASSIFIED_ROOT = os.path.join(PROJECT_ROOT, "src", "models")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "checkpoints", "best_mil_v12_seed42.pth")

# 从项目根目录导入解释模型定义
from src.explain.explain_with_offset import ResNetMIL_Explainer, FIXED_WIDTH, WINDOW_H, WINDOW_STRIDE, MAX_WINDOWS
from torchvision import transforms
from PIL import Image
import math


def get_families_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return sorted(df['family'].dropna().unique().tolist())


def resolve_map_path(map_fam_dir, img_name):
    stem = img_name.replace('.png', '')
    candidates = [
        os.path.join(map_fam_dir, f"{stem}_map.txt"),
        os.path.join(map_fam_dir, f"{stem}_offset_map.txt")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def get_top_section_distribution(img_path, map_path, model, device):
    """返回单张图片的Top-1关注区域的Section占比字典"""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    
    global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    windows_pil = [global_resize(img)]
    window_coords = [(0, orig_h)]

    img_resized = img
    new_h = orig_h
    if orig_w != FIXED_WIDTH:
        new_h = int(orig_h * FIXED_WIDTH / orig_w)
        img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)

    if new_h <= WINDOW_H:
        canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
        canvas.paste(img_resized, (0, 0))
        windows_pil.append(canvas)
        window_coords.append((0, orig_h))
    else:
        num_steps = min(math.ceil((new_h - WINDOW_H) / WINDOW_STRIDE) + 1, MAX_WINDOWS - 1)
        for i in range(num_steps):
            top_resized = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
            bottom_resized = top_resized + WINDOW_H
            windows_pil.append(img_resized.crop((0, top_resized, FIXED_WIDTH, bottom_resized)))
            window_coords.append((int(top_resized * (orig_h / new_h)), int(bottom_resized * (orig_h / new_h))))

    tensors = [normalize(to_tensor(w)) for w in windows_pil]
    tensors = torch.stack(tensors).unsqueeze(0).to(device)
    masks = torch.ones(1, len(windows_pil), dtype=torch.bool).to(device)

    with torch.no_grad():
        _, attn_weights = model(tensors, masks)
        attn_weights = attn_weights.cpu().numpy()[0]
        
    # 取局部窗口中权重最高的
    local_attn = attn_weights[1:]
    if len(local_attn) == 0: return {}
    
    best_local_idx = np.argmax(local_attn) + 1 
    top_orig, bottom_orig = window_coords[best_local_idx]
    
    # 优化：用 chunks 分块读取，而不是一次性读整个文件
    try:
        # 只读取必要的列，跳过不需要的行
        df = pd.read_csv(
            map_path,
            usecols=['pixel_y', 'section'],
            dtype={'pixel_y': np.int32, 'section': str},
            on_bad_lines='skip',
            engine='c'  # 改用 C 引擎，速度更快
        )
    except Exception as e:
        print(f"⚠️ 读取 {map_path} 失败: {e}")
        return {}
    
    if len(df) == 0 or 'pixel_y' not in df.columns or 'section' not in df.columns:
        return {}
    
    # 快速过滤
    valid_pixels = df[(df['pixel_y'] >= top_orig) & (df['pixel_y'] <= bottom_orig) & (df['section'] != 'PAD')]
    
    if len(valid_pixels) == 0: 
        return {}
    
    counts = valid_pixels['section'].value_counts(normalize=True).to_dict()
    return counts

def analyze_all_families():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    families = get_families_from_csv(TRAIN_CSV)
    num_classes = len(families)

    # 允许用环境变量做限量调试（避免全量跑太久）
    # 例如：MAX_FAMILIES=1 MAX_SAMPLES_PER_FAMILY=2 python explain/explain.py
    max_families = int(os.environ.get("MAX_FAMILIES", "0") or "0")
    max_samples_per_family = int(os.environ.get("MAX_SAMPLES_PER_FAMILY", "0") or "0")
    
    # 加载模型
    model = ResNetMIL_Explainer(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 数据路径（images和offset_maps分别存储）
    images_base_dir = os.path.join(DATA_ROOT, "images")
    offset_maps_base_dir = os.path.join(DATA_ROOT, "offset_maps")
    
    # 我们只关注最常见的几个宏观 Section，其他的归为 'Other'
    MAIN_SECTIONS = ['.text', '.rdata', '.data', '.rsrc', 'PE_HEADER']
    
    results = []

    for fi, family in enumerate(families):
        if max_families > 0 and fi >= max_families:
            break
        img_fam_dir = os.path.join(images_base_dir, family)
        map_fam_dir = os.path.join(offset_maps_base_dir, family)
        if not os.path.exists(img_fam_dir) or not os.path.exists(map_fam_dir): continue
        
        # 找这个家族下所有的 img 和 map 配对
        img_files = [f for f in os.listdir(img_fam_dir) if f.endswith('.png')]
        
        print(f"\nProcessing Family: {family} ({len(img_files)} samples)")
        family_stats = {sec: 0.0 for sec in MAIN_SECTIONS + ['Other']}
        valid_samples = 0
        
        for si, img_name in enumerate(tqdm(img_files, leave=False)):
            if max_samples_per_family > 0 and si >= max_samples_per_family:
                break
            img_path = os.path.join(img_fam_dir, img_name)
            map_path = resolve_map_path(map_fam_dir, img_name)
            
            if not map_path:
                continue
            
            # 获取单样本结果
            sec_dist = get_top_section_distribution(img_path, map_path, model, device)
            if not sec_dist: continue
            
            valid_samples += 1
            # 累加到 family_stats
            for sec, pct in sec_dist.items():
                if sec in MAIN_SECTIONS:
                    family_stats[sec] += pct
                else:
                    family_stats['Other'] += pct
                    
        # 计算平均占比
        if valid_samples > 0:
            row = {'Family': family}
            for sec in family_stats:
                row[sec] = (family_stats[sec] / valid_samples) * 100
            results.append(row)

    # 1. 保存为 CSV 数据表格
    if len(results) == 0:
        print("⚠️  No valid results collected. Please check data paths and offset_map files.")
        return
    
    df_results = pd.DataFrame(results).set_index('Family')
    out_csv = os.path.join(CLASSIFIED_ROOT, "family_section_analysis_6class.csv")
    df_results.to_csv(out_csv)
    print(f"\n✅ 数据已保存至 {out_csv}")
    
    # 2. 绘制漂亮的学术论文热力图 (Heatmap)
    plt.figure(figsize=(10, 6))

    if sns is not None:
        sns.heatmap(
            df_results,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar_kws={'label': 'Attention Focus Percentage (%)'}
        )
    else:
        # seaborn 不存在时，使用 matplotlib 手动画热力图 + 标注数值
        ax = plt.gca()
        data = df_results.values.astype(float)
        im = ax.imshow(data, aspect='auto', cmap='Blues')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Focus Percentage (%)')

        ax.set_xticks(range(len(df_results.columns)))
        ax.set_yticks(range(len(df_results.index)))
        ax.set_xticklabels(df_results.columns, rotation=45, ha='right')
        ax.set_yticklabels(df_results.index)

        # 在每个格子上标注数值
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=8)

        # 和 seaborn 类似的视觉标题
        ax.set_title("Model Attention Distribution across PE Sections by Malware Family")
        ax.set_ylabel("Malware Family")
        ax.set_xlabel("PE Binary Section")

    plt.tight_layout()
    out_png = os.path.join(CLASSIFIED_ROOT, "family_attention_heatmap_6class.png")
    plt.savefig(out_png, dpi=300)
    print(f"✅ 热力图已保存至 {out_png}")

if __name__ == "__main__":
    analyze_all_families()