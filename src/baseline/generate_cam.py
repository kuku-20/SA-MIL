import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数
# ==========================================
_BASE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_BASE, "..", ".."))
MODEL_WEIGHT_PATH = os.path.join(_PROJ_ROOT, "results", "checkpoints", "best_baseline_resnet18.pth")
TRAIN_CSV = os.path.join(_PROJ_ROOT, "data", "malware_images", "train.csv")
TARGET_IMAGE_PATH = os.path.join(_PROJ_ROOT, "data", "malware_images", "images", "GuLoader", "ee5fd99c7fc747944fc82337291094b744fca1f9f6328551ba0f10a9f025ad52.png")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ==========================================
# 2. Grad-CAM 提取器
# ==========================================
class ResNetGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        self.model.zero_grad()
        target = logits[0, class_idx]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, class_idx

# ==========================================
# 3. 主函数
# ==========================================
def main():
    # 检查权重文件是否存在
    if not os.path.exists(MODEL_WEIGHT_PATH):
        print(f"❌ 模型权重文件不存在: {MODEL_WEIGHT_PATH}")
        return
    
    # 检查图片是否存在
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"❌ 目标图片不存在: {TARGET_IMAGE_PATH}")
        return
    
    df = pd.read_csv(TRAIN_CSV)
    classes = sorted(df['family'].unique())
    num_classes = len(classes)
    
    print(f"[INFO] 类别数: {num_classes}, 类别: {classes}")
    
    # 加载模型
    model = models.resnet18(pretrained=False)
    num_classes = len(classes)
    # 改成和训练时一样的结构：Sequential with Dropout
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
    print(f"[INFO] 模型权重加载成功: {MODEL_WEIGHT_PATH}")
        
    model = model.to(DEVICE)
    
    cam_extractor = ResNetGradCAM(model, model.layer4[-1])

    # 处理图片
    original_img = Image.open(TARGET_IMAGE_PATH).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img_tensor = transform(original_img)
    img_tensor = img_tensor.repeat(3, 1, 1).unsqueeze(0).to(DEVICE)

    # 生成热力图
    cam, pred_class_idx = cam_extractor(img_tensor)
    pred_class_name = classes[pred_class_idx]
    print(f"[INFO] 预测类别: {pred_class_name} (Index: {pred_class_idx})")

    # 可视化
    img_show = original_img.resize((224, 224))
    img_show = np.array(img_show.convert('RGB'))

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_show, 0.5, heatmap, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_show)
    axes[0].set_title("Resized Input (224x224)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (Pred: {pred_class_name})", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(_PROJ_ROOT, "results", "figures", "baseline_gradcam_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 成功！热力图已保存到: {save_path}")

if __name__ == "__main__":
    main()
