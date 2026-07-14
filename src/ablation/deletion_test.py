
import os
import sys


# Project root (auto-detected)

from pathlib import Path

_PROJ_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.mil_v12 import ResNetMIL, SafeTransformerLayer, FIXED_WIDTH, WINDOW_H, WINDOW_STRIDE, MAX_WINDOWS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import cv2


# ==========================================
# 1. 核心配置与路径
# ==========================================
BASELINE_WEIGHT_PATH = str(_PROJ_DIR / "results" / "checkpoints" / "best_baseline_resnet18.pth")
SAMIL_WEIGHT_PATH = str(_PROJ_DIR / "results" / "checkpoints" / "best_mil_v12_seed42.pth")

TEST_CSV = str(_PROJ_DIR / "data" / "malware_images" / "test.csv")
IMAGES_DIR = str(_PROJ_DIR / "data" / "malware_images" / "images")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_SAMPLES = 100 

FAMILIES = ['Gozi', 'GuLoader', 'Heodo', 'IcedID', 'Trickbot', 'njrat']
NUM_CLASSES = len(FAMILIES)
CLASS_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]

# ==========================================
# 2. 传统 Baseline (ResNet) 遮挡
# ==========================================
class ResNetGradCAM_Occlusion:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.layer4[-1]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_cam(self, x, class_idx):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        orig_prob = F.softmax(logits, dim=1)[0, class_idx].item()
        
        target = logits[0, class_idx]
        target.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-8)
        return orig_prob, cam

    def evaluate_drops(self, img_path, class_idx, ratios):
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(DEVICE)
        
        orig_prob, cam = self.get_cam(x, class_idx)
        cam_resized = cv2.resize(cam, (224, 224))
        
        drops = {}
        for r in ratios:
            threshold = np.percentile(cam_resized, 100 * (1 - r))
            mask = (cam_resized >= threshold).astype(np.float32) 
            mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            x_masked = x * (1 - mask_tensor)
            
            with torch.no_grad():
                logits_masked = self.model(x_masked)
                masked_prob = F.softmax(logits_masked, dim=1)[0, class_idx].item()
                
            drops[r] = orig_prob - masked_prob
            
        return orig_prob, drops

# ==========================================
# 3. SA-MIL 模型与连带切窗遮挡
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
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  
        
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights

def extract_windows_from_image(img_path):
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    
    global_resize = transforms.Resize((WINDOW_H, FIXED_WIDTH))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    windows_pil = [global_resize(img)]
    
    img_resized = img
    new_h = orig_h
    if orig_w != FIXED_WIDTH:
        new_h = int(orig_h * FIXED_WIDTH / orig_w)
        img_resized = img.resize((FIXED_WIDTH, new_h), Image.BILINEAR)
    
    if new_h <= WINDOW_H:
        canvas = Image.new('RGB', (FIXED_WIDTH, WINDOW_H), 0)
        canvas.paste(img_resized, (0, 0))
        windows_pil.append(canvas)
    else:
        num_steps = min(math.ceil((new_h - WINDOW_H) / WINDOW_STRIDE) + 1, MAX_WINDOWS - 1)
        for i in range(num_steps):
            top_resized = min(i * WINDOW_STRIDE, new_h - WINDOW_H)
            bottom_resized = top_resized + WINDOW_H
            windows_pil.append(img_resized.crop((0, top_resized, FIXED_WIDTH, bottom_resized)))
            
    tensors = [normalize(to_tensor(w)) for w in windows_pil]
    windows_tensor = torch.stack(tensors).unsqueeze(0).to(DEVICE)
    masks = torch.ones(1, len(windows_pil), dtype=torch.bool).to(DEVICE)
    
    return windows_tensor, masks, len(windows_pil)

def evaluate_samil_drops(samil_model, img_path, class_idx, ratios):
    samil_model.eval()
    windows_tensor, masks_tensor, num_windows = extract_windows_from_image(img_path)
    
    with torch.no_grad():
        output = samil_model(windows_tensor, masks_tensor)
        if isinstance(output, tuple): logits, attn_weights = output
        else: logits, attn_weights = output, samil_model.attn_weights  
            
        orig_prob = F.softmax(logits, dim=1)[0, class_idx].item()
        
        local_attn = attn_weights[0, 1:].cpu().numpy() 
        sorted_local_indices = np.argsort(local_attn)[::-1]
        
        drops = {}
        for r in ratios:
            num_to_mask = max(1, int(len(local_attn) * r)) 
            indices_to_mask = sorted_local_indices[:num_to_mask]
            
            # 使用高斯随机噪声代替 0.0，防止模型把纯黑当作 Padding
            windows_masked = windows_tensor.clone()
            noise_tensor = torch.randn_like(windows_masked[0, 0]) * 0.5 
            
            # 【核心杀招】：强行抹黑全局窗口，逼迫模型只能看剩余的局部碎片
            windows_masked[0, 0] = noise_tensor
            
            # 【终极猛药】：块级连带摧毁 (摧毁该窗口及相邻重叠的冗余窗口)
            for idx in indices_to_mask:
                # 摧毁本尊
                windows_masked[0, idx + 1] = noise_tensor
                # 摧毁上一相邻重叠窗口
                if idx > 0: 
                    windows_masked[0, idx] = noise_tensor
                # 摧毁下一相邻重叠窗口
                if idx + 2 < num_windows: 
                    windows_masked[0, idx + 2] = noise_tensor
                
            output_masked = samil_model(windows_masked, masks_tensor)
            if isinstance(output_masked, tuple): logits_masked, _ = output_masked
            else: logits_masked = output_masked
                
            masked_prob = F.softmax(logits_masked, dim=1)[0, class_idx].item()
            drops[r] = orig_prob - masked_prob
            
    return orig_prob, drops

# ==========================================
# 4. 主执行流
# ==========================================
def main():
    print("Loading Baseline ResNet18...")
    baseline_model = models.resnet18(num_classes=NUM_CLASSES)
    baseline_model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(baseline_model.fc.in_features, NUM_CLASSES)
    )
    baseline_model.load_state_dict(torch.load(BASELINE_WEIGHT_PATH, map_location=DEVICE))
    baseline_model = baseline_model.to(DEVICE)
    baseline_evaluator = ResNetGradCAM_Occlusion(baseline_model)
    
    print("Loading SA-MIL...")
    samil_model = ResNetMIL_Vis(NUM_CLASSES).to(DEVICE)
    samil_model.load_state_dict(torch.load(SAMIL_WEIGHT_PATH, map_location=DEVICE))
    samil_model = samil_model.to(DEVICE)
    
    df = pd.read_csv(TEST_CSV).sample(NUM_SAMPLES, random_state=42)
    
    baseline_drop_stats = {r: [] for r in RATIOS}
    samil_drop_stats = {r: [] for r in RATIOS}
    
    print(f"Running Confidence Deletion Test on {NUM_SAMPLES} samples...")
    for _, row in tqdm(df.iterrows(), total=NUM_SAMPLES):
        img_path = os.path.join(IMAGES_DIR, row['family'], f"{row['sha256']}.png")
        if not os.path.exists(img_path): continue
        target_class = CLASS_TO_IDX[row['family']]
        
        _, b_drops = baseline_evaluator.evaluate_drops(img_path, target_class, RATIOS)
        for r in RATIOS: baseline_drop_stats[r].append(b_drops[r])
        
        _, s_drops = evaluate_samil_drops(samil_model, img_path, target_class, RATIOS)
        for r in RATIOS: samil_drop_stats[r].append(s_drops[r])
        
    print("\n" + "="*70)
    print("【块级连带遮挡 - 置信度暴跌测试】")
    print("-" * 70)
    print(f"{'Occlusion Ratio':<20} | {'Baseline Confidence Drop':<25} | {'SA-MIL Confidence Drop':<25}")
    print("-" * 70)
    for r in RATIOS:
        b_mean = np.mean(baseline_drop_stats[r])
        s_mean = np.mean(samil_drop_stats[r])
        # 强制格式化为百分比，保留两位小数
        print(f"Top {int(r*100)}% {' ':<12} | {b_mean*100:>7.2f}% {' ':<16} | {s_mean*100:>7.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
