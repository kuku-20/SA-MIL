
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib


# Project root (auto-detected)

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

matplotlib.use("Agg")  # 避免无 GUI 环境下 Matplotlib 初始化失败
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

# 导入你原文件中的配置和模型
# 确保 StructureAwareMIL_v12.py 在同一目录下
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.mil_v12 import ResNetMIL, SafeTransformerLayer, FIXED_WIDTH, WINDOW_H, WINDOW_STRIDE, MAX_WINDOWS, NUM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX

# 脚本所在目录（用于拼接模型/输出路径，避免相对路径依赖当前工作目录）
BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. 提取带注意力权重的模型 (继承重写 forward)
# ==========================================
class ResNetMIL_Vis(ResNetMIL):
    def forward(self, windows, masks):
        B, N, c, h, w = windows.shape
        x = self.features(windows.view(B * N, c, h, w))
        x = self.avgpool(x).flatten(1)
        x = self.proj(x).view(B, N, -1)
        
        x = x + self.pos_embed[:, :N, :]
        padding_mask = ~masks
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        attn_scores = self.attention(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~masks, -1e9)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1) # <--- 核心：提取注意力权重
        
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights

# ==========================================
# 2. 模拟数据集的切窗逻辑并记录物理坐标
# ==========================================
def get_windows_and_coords(img):
    w, h = img.size
    img_resized = img
    new_h = h
    if w != FIXED_WIDTH:
        new_h = int(h * FIXED_WIDTH / w)
        img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)

    windows = [img_resized.resize((FIXED_WIDTH, WINDOW_H))] # W0: 全局
    coords = [None] # 全局窗口不画在局部图上
    
    local_max_windows = MAX_WINDOWS - 1
    max_cover_h = WINDOW_STRIDE * (local_max_windows - 1) + WINDOW_H

    if new_h <= max_cover_h:
        # 顺序滑动
        for i in range(local_max_windows):
            top = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
            windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            coords.append((top, top + WINDOW_H))
    else:
        # 信息熵驱动显著性采样 (完全复刻 v12 逻辑)
        img_np = np.array(img_resized)
        row_scores = np.sum(img_np[:, :, 1:3], axis=(1, 2)) 
        
        search_stride = 32
        candidate_windows = []
        for top in range(0, new_h - WINDOW_H + 1, search_stride):
            window_score = np.sum(row_scores[top : top + WINDOW_H])
            candidate_windows.append((window_score, top))
            
        candidate_windows.sort(key=lambda x: x[0], reverse=True)
        
        selected_tops = [0] # 强制加头部
        min_distance = WINDOW_H // 2
        
        for score, top in candidate_windows:
            if len(selected_tops) >= local_max_windows: break
            if all(abs(top - sel_top) >= min_distance for sel_top in selected_tops):
                selected_tops.append(top)
                
        while len(selected_tops) < local_max_windows:
            fallback_top = max(0, new_h - WINDOW_H - (local_max_windows - len(selected_tops)) * search_stride)
            selected_tops.append(fallback_top)

        selected_tops.sort() # 时序排序
        
        for top in selected_tops[:local_max_windows]:
            windows.append(img_resized.crop((0, top, FIXED_WIDTH, top + WINDOW_H)))
            coords.append((top, top + WINDOW_H))

    # 将坐标还原到原图比例
    scale = h / new_h if w != FIXED_WIDTH else 1.0
    final_coords = []
    for c in coords:
        if c is None:
            final_coords.append(None)
        else:
            final_coords.append((int(c[0] * scale), int(c[1] * scale)))
            
    return windows, final_coords

# ==========================================
# 3. 主可视化函数
# ==========================================
def visualize_attention(image_path, model_path, true_label_idx=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 先加载 checkpoint，再从 classifier 权重推断类别数
    # 这样无需依赖 StructureAwareMIL_v12.py 的全局 NUM_CLASSES（import 时它仍是 0）。
    checkpoint = torch.load(model_path, map_location="cpu")
    num_classes = None
    # classifier 是 Sequential(Dropout, Linear)，其权重 key 通常为 classifier.1.weight
    for k, v in checkpoint.items():
        if k.endswith("classifier.1.weight"):
            num_classes = int(v.shape[0])
            break

    if num_classes is None:
        raise RuntimeError(f"无法从 checkpoint 推断 num_classes: 未找到 classifier.1.weight，keys={list(checkpoint.keys())[:10]}")

    model = ResNetMIL_Vis(num_classes).to(device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # 预处理
    img = Image.open(image_path).convert("RGB")
    windows, coords = get_windows_and_coords(img)
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    tensors = torch.stack([normalize(to_tensor(w)) for w in windows]).unsqueeze(0).to(device)
    masks = torch.ones(1, len(windows), dtype=torch.bool).to(device)
    
    # 前向推理提取权重
    with torch.no_grad():
        logits, attn_weights = model(tensors, masks)
        pred_label = logits.argmax(1).item()
        attn = attn_weights[0].cpu().numpy()
    
    pred_class_name = IDX_TO_CLASS[pred_label] if IDX_TO_CLASS else f"Class {pred_label}"
    print(f"预测分类: {pred_class_name}")
    print(f"各窗口注意力权重: {np.round(attn, 3)}")

    # 画图
    img_copy = img.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 150)) # 压暗背景突出热力
    img_with_overlay = Image.alpha_composite(img_copy, overlay)
    draw = ImageDraw.Draw(img_with_overlay)
    
    colormap = cm.get_cmap('Reds') # 使用红色热力图映射
    
    # 找到局部窗口中的最大权重用于归一化颜色显示 (跳过全局 W0)
    local_attn = attn[1:] 
    max_attn = max(local_attn) if len(local_attn) > 0 else 1.0
    
    for i in range(1, len(coords)): # 跳过 W0 (全局窗口)
        top, bottom = coords[i]
        weight = attn[i]
        
        # 计算颜色强度 (相对局部最大值归一化)
        intensity = weight / (max_attn + 1e-9)
        color = colormap(intensity)
        r, g, b = [int(x * 255) for x in color[:3]]
        
        # 权重越高，透明度越低，颜色越红
        alpha = int(100 + 155 * intensity) 
        
        # 填充热力颜色
        draw.rectangle([0, top, img.size[0], bottom], fill=(r, g, b, alpha))
        # 边框
        draw.rectangle([0, top, img.size[0], bottom], outline=(255, 0, 0, 255), width=4)
        
        # 标注 W_i 和 权重
        text = f"W{i}: {weight:.3f}"
        text_y = top + 10 if top + 10 < img.size[1] else top - 30
        draw.text((10, text_y), text, fill="white", stroke_width=2, stroke_fill="black")

    # Matplotlib 拼图展示
    fig, axes = plt.subplots(1, 2, figsize=(10, 16))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img_with_overlay)
    axes[1].set_title(f"Attention Heatmap\nPred: {pred_class_name}", fontsize=14, color='darkred')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = str(BASE_DIR / "attention_verification.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 注意力验证图已保存至: {save_path}")

if __name__ == "__main__":
    # ========================================================
    # ⚠️ 1. 填入你刚才训练出的最优权重路径
    MODEL_PATH = str(BASE_DIR / "best_mil_v12_seed42.pth")
    
    # ⚠️ 2. 挑一张超大的 GuLoader 或 Trickbot 图片
TEST_IMAGE_PATH = str(_PROJ_DIR / "data" / "malware_images" / "images" / "GuLoader" / "ee5fd99c7fc747944fc82337291094b744fca1f9f6328551ba0f10a9f025ad52.png")
    # ========================================================
    
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_IMAGE_PATH):
        visualize_attention(TEST_IMAGE_PATH, MODEL_PATH)
    else:
        print("请检查 MODEL_PATH 和 TEST_IMAGE_PATH 是否存在！")
