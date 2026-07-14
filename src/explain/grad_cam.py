
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms

# 假设你已经有定义好的模型并加载了权重
# from your_model_file import ResNetMIL 

def run_micro_traceback(model, image_path, offset_map_path, target_window_index=7, window_h=224, window_w=256):
    print(f"🔬 开启微观追踪 (Micro-Traceback) - Window #{target_window_index}")
    
    # 1. 读取原图并切出目标 Window
    img = Image.open(image_path).convert('RGB')
    
    # 计算该窗口的 Y 坐标范围
    y_start = target_window_index * window_h
    y_end = y_start + window_h
    
    # 裁剪出目标窗口图像 (256x224)
    window_img = img.crop((0, y_start, window_w, y_end))
    
    # 图像预处理 (和你的 Dataset 里一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(window_img).unsqueeze(0) # [1, 3, 224, 256]
    
    # 2. 配置 Grad-CAM
    # 你的模型使用了特征提取器，我们取 ResNet 的最后一层卷积层
    # 假设你的 self.feature_extractor 是 ResNet
    target_layers = [model.feature_extractor.layer4[-1]] 
    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    # 生成热力图 (224x256 的二维矩阵，值在 0~1 之间)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # ---- 3. 提取最高光像素坐标 (Top 1%) ----
    threshold = np.percentile(grayscale_cam, 99) # 取最热的 1%
    hot_y, hot_x = np.where(grayscale_cam >= threshold)
    
    # 将窗口内的局部坐标，映射回原图的全局 Y 坐标
    global_hot_y = hot_y + y_start
    
    print(f"🔥 发现 {len(hot_x)} 个核心激活性像素 (Top 1% 热度)！")
    
    # ---- 4. 反查物理偏移映射表 ----
    print(f"📄 读取 Offset Map...")
    df = pd.read_csv(offset_map_path, low_memory=False)
    
    # 过滤掉 PAD
    df = df[df['file_offset'] != 'PAD'].copy()
    df['file_offset'] = pd.to_numeric(df['file_offset'])
    
    exact_offsets = []
    for x, y in zip(hot_x, global_hot_y):
        match = df[(df['pixel_x'] == x) & (df['pixel_y'] == y)]
        if not match.empty:
            exact_offsets.append(match['file_offset'].values[0])
            
    if exact_offsets:
        min_offset = min(exact_offsets)
        max_offset = max(exact_offsets)
        
        print("="*60)
        print(f"🎯 微观定位成功！从 57KB 缩小到 {len(exact_offsets)} Bytes！")
        print(f"🔢 精确物理文件偏移 (Hex): {hex(min_offset)} -> {hex(max_offset)}")
        print("="*60)
        
        # 5. 可视化保存热力图（用于论文插图）
        img_float = np.float32(window_img) / 255
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save(f"Window_{target_window_index}_GradCAM.png")
        print(f"🖼️ 微观热力图已保存为 Window_{target_window_index}_GradCAM.png")
    else:
        print("⚠️ 未能在映射表中找到对应的非 Padding 偏移。")

# ================= 使用示例 =================
# run_micro_traceback(
#     model=your_trained_model, 
#     image_path="8e5d...0f21.png", 
#     offset_map_path="8e5d...0f21_map.txt", 
#     target_window_index=7
# )
