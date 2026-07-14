import os
import csv
import math
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ===== 路径配置（对齐你当前新数据） =====
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "malware_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "checkpoints", "best_mil_v12_seed42.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures", "gradcam_samples")

# ===== 模型窗口参数（必须与训练保持一致） =====
FIXED_WIDTH = 256
WINDOW_H = 224
WINDOW_STRIDE = 112
MAX_WINDOWS = 12

# 导入解释模型
from src.explain.explain_with_offset import ResNetMIL_Explainer


def build_class_mapping(csv_path):
    families = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            families.add(row['family'])
    families = sorted(families)
    class_to_idx = {fam: i for i, fam in enumerate(families)}
    return families, class_to_idx


def collect_family_images(images_root, families):
    fam_to_images = {}
    for fam in families:
        fam_dir = os.path.join(images_root, fam)
        if not os.path.isdir(fam_dir):
            fam_to_images[fam] = []
            continue
        imgs = [os.path.join(fam_dir, x) for x in os.listdir(fam_dir) if x.lower().endswith('.png')]
        fam_to_images[fam] = sorted(imgs)
    return fam_to_images


def extract_windows_with_coords(img):
    """
    复现训练时切窗逻辑，并返回每个局部窗口在原图中的 y 区间。
    返回:
      windows_pil: [global, local1, local2, ...]
      local_ranges: [(start_y, end_y), ...]  # 对应 local1/local2...
    """
    orig_w, orig_h = img.size
    global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))

    windows_pil = [global_resize(img)]
    local_ranges = []

    img_resized = img
    new_h = orig_h
    if orig_w != FIXED_WIDTH:
        new_h = int(orig_h * FIXED_WIDTH / orig_w)
        img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)

    if new_h <= WINDOW_H:
        canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
        canvas.paste(img_resized, (0, 0))
        windows_pil.append(canvas)
        local_ranges.append((0, orig_h))
    else:
        num_steps = min(math.ceil((new_h - WINDOW_H) / WINDOW_STRIDE) + 1, MAX_WINDOWS - 1)
        for i in range(num_steps):
            top_resized = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
            bottom_resized = top_resized + WINDOW_H
            windows_pil.append(img_resized.crop((0, top_resized, FIXED_WIDTH, bottom_resized)))

            top_orig = int(top_resized * (orig_h / new_h))
            bottom_orig = int(bottom_resized * (orig_h / new_h))
            local_ranges.append((top_orig, bottom_orig))

    return windows_pil, local_ranges


def infer_attention_weights(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    windows_pil, local_ranges = extract_windows_with_coords(img)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    tensors = [normalize(to_tensor(w)) for w in windows_pil]
    tensors = torch.stack(tensors).unsqueeze(0).to(device)
    masks = torch.ones(1, len(windows_pil), dtype=torch.bool).to(device)

    with torch.no_grad():
        logits, attn_weights = model(tensors, masks)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred_idx].item())
        attn_weights = attn_weights[0].detach().cpu().numpy()

    # attn[0] 是 global，叠加时只用 local
    local_attn = attn_weights[1:] if len(attn_weights) > 1 else np.array([])
    return local_attn, local_ranges, pred_idx, pred_conf


def generate_stitched_heatmap(original_image_path, local_attention_weights, local_ranges, save_path):
    """
    将局部窗口 attention 按原图 y 区间反向拼接，并与 RGB 图叠加。
    """
    img_bgr = cv2.imread(original_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到图像: {original_image_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape

    if img_w != FIXED_WIDTH:
        img = cv2.resize(img, (FIXED_WIDTH, img_h))
        img_h, img_w, _ = img.shape

    heatmap_mask = np.zeros((img_h, img_w), dtype=np.float32)
    overlap_count = np.zeros((img_h, img_w), dtype=np.float32)

    n = min(len(local_attention_weights), len(local_ranges))
    for i in range(n):
        start_y, end_y = local_ranges[i]
        start_y = max(0, min(start_y, img_h - 1))
        end_y = max(start_y + 1, min(end_y, img_h))

        heatmap_mask[start_y:end_y, :] += float(local_attention_weights[i])
        overlap_count[start_y:end_y, :] += 1.0

    overlap_count[overlap_count == 0] = 1.0
    heatmap_mask = heatmap_mask / overlap_count

    heatmap_mask = (heatmap_mask - np.min(heatmap_mask)) / (np.max(heatmap_mask) - np.min(heatmap_mask) + 1e-8)
    heatmap_mask_u8 = np.uint8(255 * heatmap_mask)

    heatmap_color = cv2.applyColorMap(heatmap_mask_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    alpha = 0.45
    overlay_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    plt.figure(figsize=(10, max(4, img_h / 120)), dpi=250)
    plt.subplot(1, 2, 1)
    plt.title("Original RGB", fontsize=12)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Attention Overlay", fontsize=12)
    plt.imshow(overlay_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.08)
    plt.close()


def run_batch_demo(samples_per_family=2, seed=42):
    random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    families, class_to_idx = build_class_mapping(TRAIN_CSV)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNetMIL_Explainer(len(families)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    images_root = os.path.join(DATA_ROOT, "images")
    fam_to_images = collect_family_images(images_root, families)

    total = 0
    for fam in families:
        candidates = fam_to_images.get(fam, [])
        if not candidates:
            print(f"[WARN] {fam}: 无可用图片")
            continue

        k = min(samples_per_family, len(candidates))
        chosen = random.sample(candidates, k)

        out_fam_dir = os.path.join(OUTPUT_DIR, fam)
        os.makedirs(out_fam_dir, exist_ok=True)

        print(f"\nFamily={fam} | pick {k}/{len(candidates)}")
        for p in chosen:
            local_attn, local_ranges, pred_idx, pred_conf = infer_attention_weights(model, p, device)
            stem = os.path.splitext(os.path.basename(p))[0]
            save_path = os.path.join(out_fam_dir, f"{stem}_overlay.png")
            generate_stitched_heatmap(p, local_attn, local_ranges, save_path)

            pred_name = idx_to_class.get(pred_idx, str(pred_idx))
            print(f"  [+] {stem} -> pred={pred_name} ({pred_conf:.4f}) | {save_path}")
            total += 1

    print(f"\n✅ 完成：共生成 {total} 张叠加图，输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_batch_demo(samples_per_family=2, seed=42)
