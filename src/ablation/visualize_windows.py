
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ==========================================
# 配置参数 (与你的 v12 保持完全一致)
# ==========================================


# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

FIXED_WIDTH   = 256
WINDOW_H      = 224
WINDOW_STRIDE = 112
MAX_WINDOWS   = 12

def get_window_coordinates(img):
    """
    运行你的 v12 切窗逻辑，但只返回坐标 (y_start, y_end)，不实际裁剪图片
    用于在原图上画框。
    """
    w, h = img.size
    
    # 记录坐标的列表，格式为 (y_start, y_end)
    coordinates = []
    
    # Window 0 是全局缩放，这里为了在原图展示局部框，我们跳过 Window 0 的可视化
    # 我们只关注剩下的 11 个局部高清窗口的落点
    local_max_windows = MAX_WINDOWS - 1 # 11
    
    new_h = int(h * FIXED_WIDTH / w) if w != FIXED_WIDTH else h
    max_cover_h = WINDOW_STRIDE * (local_max_windows - 1) + WINDOW_H

    if new_h <= max_cover_h:
        # 99% 的正常文件：顺序滑动
        for i in range(local_max_windows):
            top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
            coordinates.append((top, top + WINDOW_H))
    else:
        # 1% 的超大文件：首尾拼接 (Head 5 + Tail 6)
        head_windows = 5
        tail_windows = local_max_windows - head_windows # 6
        
        # 头部 5 个
        for i in range(head_windows):
            top = i * WINDOW_STRIDE
            coordinates.append((top, top + WINDOW_H))
        
        # 尾部 6 个
        for i in range(tail_windows - 1, -1, -1):
            top = new_h - WINDOW_H - i * WINDOW_STRIDE
            coordinates.append((top, top + WINDOW_H))

    # 还原回原始图像的高度比例
    scale = h / new_h if w != FIXED_WIDTH else 1.0
    original_coords = [(int(top * scale), int(bottom * scale)) for top, bottom in coordinates]
    
    return original_coords

def visualize_sampling(image_path, save_path="sampling_visualization.png"):
    # 1. 打开原图
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # 2. 获取切窗的物理坐标
    coords = get_window_coordinates(img)
    
    # 3. 创建用于画图的对象
    # 为了让红色框更清晰，我们稍微调暗原图作为背景
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 100))
    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(img_with_overlay)
    
    # 4. 在图上画出红色的边框和半透明填充
    colors = ['#FF3333'] * len(coords)  # 红色框
    
    for i, (top, bottom) in enumerate(coords):
        # 画半透明红色填充区
        draw.rectangle([0, top, w, bottom], fill=(255, 50, 50, 40))
        # 画加粗实心红框
        draw.rectangle([0, top, w, bottom], outline="red", width=4)
        
        # 添加窗口编号文本 (比如 W1, W2 ...)
        # 防止文字超出上边界
        text_y = top + 10 if top + 10 < h else top - 30
        draw.text((10, text_y), f"W{i+1}", fill="white", stroke_width=2, stroke_fill="black")

    # 5. 使用 matplotlib 拼接展示（左边原图，右边采样图）
    fig, axes = plt.subplots(1, 2, figsize=(12, min(20, h/100))) # 动态调整高度比例
    
    axes[0].imshow(img)
    axes[0].set_title(f"Original Malware Image\n(Size: {w}x{h})", fontsize=14, pad=15)
    axes[0].axis('off')
    
    axes[1].imshow(img_with_overlay)
    
    # 如果是超大文件，计算中间跳过的“盲区”大小
    if h > (WINDOW_STRIDE * 10 + WINDOW_H):
        gap = coords[5][0] - coords[4][1]
        axes[1].set_title(f"Head-Tail Splicing\n(Skipped {gap} rows of padding)", fontsize=14, pad=15, color='darkred')
    else:
        axes[1].set_title(f"Sequential Sliding Windows\n(Fully Covered)", fontsize=14, pad=15, color='darkgreen')
        
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图已成功保存为: {save_path}")
    plt.close()

if __name__ == "__main__":
    # ========================================================
    # ⚠️ 请在这里填入一个【超大恶意软件】的图片路径
    # 建议去 data/malware_images 里面挑一个文件大小最大的 .png
    # ========================================================
TEST_IMAGE_PATH = str(_PROJ_DIR / "data" / "malware_images" / "images" / "Trickbot" / "faaa0098ad3de31c95506576653962bf783bdf347b6d22255d707561e30c5350.png")
    
    if os.path.exists(TEST_IMAGE_PATH):
        visualize_sampling(TEST_IMAGE_PATH)
    else:
        print(f"❌ 找不到图片: {TEST_IMAGE_PATH}\n请在代码底部修改为你真实的大图片路径！")
