import sys


# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import math

# 导入 v12 模型
from src.explain.explain_with_offset import ResNetMIL_Explainer, FIXED_WIDTH, WINDOW_H, WINDOW_STRIDE, MAX_WINDOWS

def analyze_sample(model, image_path, device):
    """提取单张图片中 Attention 分数最高的窗口及其像素坐标"""
    
    # 图像预处理 (与训练时完全一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
    
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # 1. 全局缩放窗口
    windows_pil = [global_resize(img)]
    window_coords = [(0, orig_h)]
    
    # 2. 局部切窗逻辑（与 explain.py 中的逻辑保持一致）
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
            top_orig = int(top_resized * (orig_h / new_h))
            bottom_orig = int(bottom_resized * (orig_h / new_h))
            window_coords.append((top_orig, bottom_orig))
    
    # 3. 转换为张量
    tensors = [transform(w) for w in windows_pil]
    windows_tensor = torch.stack(tensors).unsqueeze(0).to(device)
    masks = torch.ones(1, len(windows_pil), dtype=torch.bool).to(device)
    
    # 4. 获取模型预测与 Attention 权重
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(windows_tensor, masks)
        attn_weights_np = attn_weights.cpu().numpy()[0]
        pred_class = logits.argmax(1).item()
    
    # 5. 找到最重要的窗口（忽略全局窗口，从局部窗口中选择）
    local_attn = attn_weights_np[1:]  # 跳过全局窗口 (index 0)
    if len(local_attn) == 0:
        # 如果没有局部窗口，用全局窗口
        max_attn_idx = 0
    else:
        max_local_idx = np.argmax(local_attn)
        max_attn_idx = max_local_idx + 1
    
    max_attn_weight = attn_weights_np[max_attn_idx]
    y_start, y_end = window_coords[max_attn_idx]
    
    print("="*60)
    print(f"🔬 Traceback Analysis for: {os.path.basename(image_path)}")
    print("="*60)
    print(f"Predicted Class Index    : {pred_class}")
    print(f"Top Attention Window     : Window #{max_attn_idx} (Weight: {max_attn_weight:.4f})")
    print(f"Original Image Size      : {orig_w}x{orig_h}")
    print(f"Pixel Y-Coordinates      : {y_start} to {y_end}")
    print(f"Attention Distribution   :")
    for idx, weight in enumerate(attn_weights_np):
        coord_info = f"Global" if idx == 0 else f"Local Y:[{window_coords[idx][0]}, {window_coords[idx][1]}]"
        print(f"  Window #{idx:2d} ({coord_info:30s}): {weight:.4f}")
    print("="*60)
    
    return (y_start, y_end), max_attn_idx

if __name__ == '__main__':
    # ================= 配置区 =================
MODEL_WEIGHTS = str(_PROJ_DIR / "results" / "checkpoints" / "best_mil_v12_seed42.pth")
SAMPLE_IMAGE_PATH = str(_PROJ_DIR / "data" / "malware_images" / "images" / "Trickbot" / "faaa0098ad3de31c95506576653962bf783bdf347b6d22255d707561e30c5350.png")
    # ==========================================
    
    print("[DEBUG] 设备检测与模型初始化开始")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[DEBUG] 使用设备: {device}")
    print(f"[DEBUG] 加载模型权重: {MODEL_WEIGHTS}")
    model = ResNetMIL_Explainer(num_classes=6).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    print("[DEBUG] 模型加载完成")
    print(f"[DEBUG] 加载图片: {SAMPLE_IMAGE_PATH}")
    
    # 执行分析
    coords, window_idx = analyze_sample(model, SAMPLE_IMAGE_PATH, device)
    print("[DEBUG] analyze_sample 执行完成")
    
    # 将结果写入临时文件，供下一步脚本读取
    try:
        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.temp'))
        os.makedirs(temp_dir, exist_ok=True)
        result_path = os.path.join(temp_dir, 'trace_result.json')
        import json
        payload = {
            'image_path': os.path.abspath(SAMPLE_IMAGE_PATH),
            'y_min': int(coords[0]),
            'y_max': int(coords[1]),
            'window_index': int(window_idx),
            'model_weights': os.path.abspath(MODEL_WEIGHTS)
        }
        with open(result_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[DEBUG] 已将 traceback 结果写入: {result_path}")
    except Exception as e:
        print(f"[WARN] 无法写入临时结果: {e}")
    
    print("\n[NEXT STEP]:")
    print(f"1. Open your Offset Map for this sample.")
    print(f"2. Look up the physical file offsets corresponding to Image Y-coordinates {coords[0]} to {coords[1]}.")
    print(f"3. Note down the Hex Offsets (e.g., 0x4A00 to 0x4E00).")
    print(f"4. Open the original PE file in IDA Pro, press 'G', and jump to that offset.")
