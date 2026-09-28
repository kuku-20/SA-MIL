# 超越语义漂移：基于结构感知多实例学习的高保真字节级恶意软件归因

> 论文：*Beyond Semantic Drift: High-Fidelity Byte-Level Malware Attribution via Structure-Aware MIL*

基于 PyTorch 实现的 **结构感知多实例学习（SA-MIL）** 框架，用于细粒度 Windows PE 恶意软件家族分类，支持确定的物理字节级逆向追溯。

## 概述

基于图像的恶意软件分类面临一个根本限制：**语义漂移**（Semantic Drift）—— 二进制文件转图像过程中的空间变换（全局缩放、固定裁剪）会破坏像素与字节之间的确定对应关系。

SA-MIL 通过三个协同模块解决这一问题：

1. **结构感知语义成像（Structure-Aware Semantic Imaging）** — 将 PE 字节折叠为多通道 RGB 图像（R=原始字节、G=节区边界、B=局部信息熵），通过确定映射 `\mathcal{M}(x,y) \to \langle \text{Offset}, \text{Section} \rangle`严格保持像素与偏移量的对应关系。
2. **熵驱动实例采样（Entropy-Driven Instance Sampling）** — 通过显著性评分和一维非极大值抑制（1D-NMS, IoU=0.3）过滤低信息冗余区域，动态聚焦于最具判别力的 top-12 个实例，规避内存瓶颈。
3. **双粒度决策追溯（Dual-Granularity Decision Traceback）** — 将实例级注意力权重映射回精确的物理字节区间（$\mathcal{O}(1)$ 逆映射），提供可直接导入 IDA Pro 的可寻址归因锚点。

## 方法

### SA-MIL 框架总览

![SA-MIL 框架总览](figures/sa_mil_overview.png)

*SA-MIL 框架总览：模块 1 通过 `\mathcal{M}(x,y)`将字节折叠为 RGB 图像；模块 2 通过熵驱动采样构造 MIL 包；模块 3 将注意力逆映射到物理字节区间。*

### 模块一：结构感知语义成像

![结构感知成像](figures/structure_aware_imaging.png)

*结构感知成像：R 通道（原始字节）、G 通道（节区先验）、B 通道（局部熵），以及确定的偏移量映射。*

### 模块二与三：实例采样与 Transformer 聚合

![Transformer 流水线](figures/transformer.png)

*熵驱动采样、ResNet-50 编码和 Transformer 全局拼接流水线。*

### 逆映射：字节级归因

![逆映射](figures/inverse_mapping.png)

*注意力权重通过 `\mathcal{M}(x,y)`逆映射到连续字节区间，使分析人员可以直接检查被标记的偏移位置。*

---

### 流水线

```
PE 文件 → 多通道 RGB 图像 
        → 熵加权滑动窗口采样 + 1D-NMS 
        → ResNet-50 实例编码器 
        → 位置编码 Transformer 聚合器 
        → 门控注意力池化 → 家族分类
        → 注意力权重 → 逆偏移量映射 → 字节级归因
```

### 主要结果

| 指标 | 数值 |
|------|------|
| **Macro-F1（单模型）** | **98.80%** |
| **Macro-F1（集成 + TTA）** | **99.28%** |
| **删除测试（10% 遮挡）** | **置信度下降 22.70%** |
| 数据集 | 3,963 个 PE 样本，6 个家族（MalwareBazaar） |

SA-MIL 在所有评估的基线方法中表现最优 —— 包括 MalConv2（96.37%）、AB-MIL（95.52%）和 IMCFN（94.97%）—— 同时提供可信赖、可物理验证的归因。

## 项目结构

```
├── README.md
├── README_CN.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/                          # 数据准备脚本
│   ├── download.py                # 从 MalwareBazaar API 下载样本
│   ├── unlock.py                  # 解压密码保护的压缩包
│   ├── reorganize.py              # 按家族重新组织文件
│   ├── generate_labels.py         # 生成标签 CSV 文件
│   ├── split_and_organize.py      # 训练/验证/测试集划分 (70/15/15)
│   ├── validate_no_leak.py        # SHA-256 去重与泄漏检查
│   ├── labels/                    # 真实标签文件
│   │   ├── MalwareBazaar_Labels.csv        # 原始下载标签（3,971 样本）
│   │   └── Existing_labels.csv             # 过滤后的标签（3,963 样本，与论文一致）
│   ├── malware_images/            # 生成的图像数据集
│   │   ├── train.csv / val.csv / test.csv  # 数据划分
│   │   ├── images/                # 按家族组织的多通道 RGB 图像
│   │   └── offset_maps/           # PE 节区偏移量查找表
│   ├── malware_samples_raw/       # 原始下载的 ZIP 文件（.gitignored）
│   └── malware_samples_pe/        # 解压后的 PE 文件（.gitignored）
│
├── src/
│   ├── preprocessing/
│   │   ├── b2image.py             # PE → 多通道 RGB（+ 偏移量映射）
│   │   ├── find_offset.py         # 通过 pefile 定位 PE 节区偏移
│   │   └── file_offset.py         # 验证像素到偏移量的映射
│   │
│   ├── models/
│   │   ├── mil_v11.py             # SA-MIL v11（RandAugment + 标签平滑）
│   │   └── mil_v12.py             # SA-MIL v12（熵驱动显著性采样）
│   │
│   ├── training/
│   │   ├── evaluate.py            # 多种子集成 + TTA 评估
│   │   └── visualize.py           # 训练曲线与混淆矩阵绘图
│   │
│   ├── explain/
│   │   ├── explain_with_offset.py # 暴露注意力权重的 ResNetMIL 包装器
│   │   ├── explain.py             # 家族级节区注意力分析
│   │   ├── trace_attention.py     # 逐样本注意力 → 字节偏移量追踪
│   │   ├── grad_cam.py            # Grad-CAM（基于 pytorch-grad-cam 库）
│   │   ├── grad_cam_alt.py        # 拼接式热力图叠加生成
│   │   ├── visualize_attention.py # 原始图像上的注意力热力图
│   │   └── analysis_model.py      # 可解释性模型辅助（v11）
│   │
│   ├── baseline/
│   │   ├── malconv.py             # MalConv（Raff et al., 2017）— 一维 CNN，64KB 截断
│   │   ├── malconv2.py            # MalConv2（Raff et al., 2021）— 全字节门控卷积
│   │   ├── resnet_cnn.py          # ResNet-18 — 图像级分类（全局缩放）
│   │   ├── peer_methods.py        # 4 种对比方法：IMCFN (VGG16)、ViT-B/16、AB-MIL、TransMIL
│   │   ├── samil_fair.py          # SA-MIL 公平条件（与对比方法相同预算）
│   │   └── generate_cam.py        # 基线 Grad-CAM 对比可视化
│   │
│   └── ablation/
│       ├── channel_ablation.py    # R/G/B 通道贡献实验
│       ├── flip_v11.py            # HorizontalFlip 增强测试（v11）
│       ├── flip_v12.py            # HorizontalFlip 增强测试（v12）
│       ├── deletion_test.py       # 可控遮挡 / 删除测试（XAI 忠实度）
│       ├── deletion_curve.py      # 删除/插入指标曲线绘制
│       ├── run_pure_test.py       # 纯评估（无增强，单种子）
│       ├── visualize_windows.py   # 窗口采样策略可视化
│       ├── plot_charts.py         # 通道消融结果图表
│       └── plot_charts2.py        # 额外消融图表
│
├── results/
│   ├── checkpoints/               # 训练好的模型权重 (*.pth) — v12、基线、消融
│   ├── figures/                   # 生成的图表 (*.png, *.pdf)
│   ├── tables/                    # 结果表格 (*.csv, *.json)
│   └── reports/                   # 测试报告 (*.txt, *.docx)
│
├── configs/                       # 配置模板
│
└── docs/
    └── region_2a000_380ff.bin     # 用于可解释性演示的 PE 样本片段
```

## 数据集

**MalwareBazaar** — 3,963 个 Windows PE 样本，涵盖 6 个恶意软件家族（来自 `data/labels/Existing_labels.csv`）：

| 家族 | 总数 | 训练集 (70%) | 验证集 (15%) | 测试集 (15%) |
|------|------|-------------|-------------|-------------|
| njrat | 939 | 656 | 141 | 142 |
| Trickbot | 880 | 616 | 132 | 132 |
| Gozi | 768 | 537 | 115 | 116 |
| GuLoader | 586 | 410 | 88 | 88 |
| IcedID | 580 | 404 | 87 | 89 |
| Heodo | 210 | 148 | 29 | 33 |
| **总计** | **3,963** | **2,771** | **592** | **600** |

所有样本均经过 SHA-256 去重，防止跨集泄漏。`data/labels/MalwareBazaar_Labels.csv`（3,971 样本）为过滤前的原始下载标签。

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
# 1. 从 MalwareBazaar 下载样本
python data/download.py

# 2. 解压密码保护的压缩包
python data/unlock.py

# 3. 将 PE 转换为多通道 RGB 图像
python src/preprocessing/b2image.py \
    --input_dir data/malware_samples_pe \
    --output_dir data/malware_images/images

# 4. 生成标签并划分数据集
python data/generate_labels.py
python data/split_and_organize.py

# 5. 验证无数据泄漏
python data/validate_no_leak.py
```

## 训练

### SA-MIL v12（主模型）

旗舰模型采用基于熵驱动的显著性采样和 1D NMS 处理大文件：

```bash
python src/models/mil_v12.py
```

关键超参数（可在 `src/models/mil_v12.py` 顶部配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FIXED_WIDTH` | 256 | 图像宽度（字节折叠列数） |
| `WINDOW_H` | 224 | 滑动窗口高度 |
| `WINDOW_STRIDE` | 112 | 窗口滑动步长 |
| `MAX_WINDOWS` | 12 | 每个包的最大实例数（熵选择） |
| `HIDDEN_DIM` | 512 | Transformer 隐藏维度 |
| `SEEDS` | [42, 123, 456] (v11)、[123] (v12) | 集成用随机种子 |
| `MAX_EPOCHS` | 50 | 训练轮数 |

### SA-MIL v11

v11 基线使用 RandAugment + 标签平滑：

```bash
python src/models/mil_v11.py
```

### 基线方法

SA-MIL 与 **8 种基线方法** 进行对比，涵盖三个类别：

| 类别 | 方法 | 脚本 |
|------|------|------|
| **长序列（截断）** | MalConv (Raff et al., 2017) — 一维 CNN，64KB 截断 | `src/baseline/malconv.py` |
| **长序列（全长）** | MalConv2 (Raff et al., 2021) — 全字节门控卷积 | `src/baseline/malconv2.py` |
| **视觉（全局缩放）** | ResNet-18 — 强制 224×224 缩放 | `src/baseline/resnet_cnn.py` |
| **视觉（全局缩放）** | IMCFN (Vasan et al., 2020) — VGG16 迁移 | `src/baseline/peer_methods.py` |
| **视觉（全局缩放）** | ViT-B/16 (Dosovitskiy et al., 2021) | `src/baseline/peer_methods.py` |
| **MIL（门控注意力）** | AB-MIL (Ilse et al., 2018) | `src/baseline/peer_methods.py` |
| **MIL（CLS Token）** | TransMIL (Shao et al., 2023) | `src/baseline/peer_methods.py` |
| **MIL（公平条件）** | SA-MIL fair（与对比方法相同预算） | `src/baseline/samil_fair.py` |

```bash
python src/baseline/malconv.py
python src/baseline/malconv2.py
python src/baseline/resnet_cnn.py
python src/baseline/peer_methods.py
python src/baseline/samil_fair.py
```

### 多种子集成评估

```bash
python src/training/evaluate.py
```

加载多个训练好的检查点，运行集成与 TTA，结果保存至 `results/reports/` 和 `results/figures/`。

### 二分类与鲁棒性补充实验

仓库同时保留了用于鲁棒性分析和审稿回应的补充实验：

| 实验 | 脚本 | 关键结果 |
|------|------|----------|
| 平衡二分类 | `src/models/mil_binary.py` | AUROC 0.9962，FPR 2.53% |
| 极端不平衡（20:1） | `data_prep_imbalance.py` | AUROC 0.9772，TPR 95.56%，FPR 3.44% |
| UPX 加壳良性文件 | `exp2_final.py` | FPR 3.45% @0.5，1.45% @0.9 |
| MorphKatz 混淆良性文件 | `exp3_obfuscation.py`、`exp3_paired_inference.py` | FPR 3.10% @0.5，原始文件 3.15%；0.5 阈值下仅 3 个配对预测翻转 |

这些实验属于补充鲁棒性分析，项目主任务仍是六家族归因与可物理追溯的字节级定位。

## 多通道成像

SA-MIL 的结构感知成像将每个 PE 字节编码为三通道像素：

- **R 通道**（原始字节）：`P_R(i) = b_i` — 直接保留操作码序列和局部数据模式。
- **G 通道**（节区先验）：PE 节区拓扑映射为离散值（`.text`→32、`.rdata`→64、`.data`→128、`.rsrc`→160，其他按 RWX 权限映射）— 在节区边界引入高频边缘。
- **B 通道**（局部熵）：基于 256 字节滑动窗口的 Shannon 熵，线性缩放到 [0,255] — 突出加密/压缩载荷。

逆映射 $\mathcal{M}(x,y) \to \langle \text{Offset}, \text{Section} \rangle$ 实现 $\mathcal{O}(1)$ 分辨率，支持从注意力权重无损追溯至物理字节区间。

## 可解释性

```bash
# 家族级节区注意力分析（RQ3）
python src/explain/explain.py

# 逐样本注意力追踪 → 字节偏移量
python src/explain/trace_attention.py

# Grad-CAM 可视化（基于 pytorch-grad-cam）
python src/explain/grad_cam.py

# 替代 Grad-CAM（拼接式热力图叠加）
python src/explain/grad_cam_alt.py

# 原始图像上的注意力热力图
python src/explain/visualize_attention.py
```

### 可控删除测试（XAI 忠实度）

删除测试系统性地将 top-k% 高注意力区域置零，并测量置信度下降（参见 `src/ablation/deletion_test.py`）：

- **SA-MIL**：10% 遮挡时置信度下降 **22.70%**，50% 遮挡时下降 **33.24%**
- **基线 Grad-CAM**：50% 遮挡时仅下降 **3.63%**

## 消融研究

### 通道贡献

通道配置与测试集 Macro-F1：

| 配置 | 说明 | Macro-F1 |
|------|------|----------|
| R only | 仅原始字节（无节区先验、无熵） | 93.17% |
| RG | 字节 + 节区边界 | 95.33% |
| RB | 字节 + 局部熵 | 95.41% |
| **RGB** | **全协同** | **98.80%** |

### 增强策略

| 配置 | Macro-F1 |
|------|----------|
| 无增强 | 98.80% |
| 去除 RandAug | 98.75% |
| 去除 HorizontalFlip | 98.73% |
| **全增强流水线** | **99.28%** |

```bash
# R/G/B 通道贡献实验
python src/ablation/channel_ablation.py

# HorizontalFlip 增强效果（v12）
python src/ablation/flip_v12.py

# HorizontalFlip 增强效果（v11）
python src/ablation/flip_v11.py

# 可控遮挡 / 删除测试
python src/ablation/deletion_test.py

# 删除/插入指标曲线
python src/ablation/deletion_curve.py

# 纯评估（无增强，单种子）
python src/ablation/run_pure_test.py

# 窗口采样策略可视化
python src/ablation/visualize_windows.py
```

## 实验结果

所有实验输出组织在 `results/` 目录下：

- **检查点**（`results/checkpoints/`）：v12、resnet18 基线以及通道/翻转消融实验的训练权重 `.pth` 文件。注意：v11 权重需先运行 `src/models/mil_v11.py` 训练得到。
- **图表**（`results/figures/`）：混淆矩阵、训练曲线、注意力热力图、消融图表。
- **表格**（`results/tables/`）：分家族 CSV 分析、对比方法 JSON 结果。
- **报告**（`results/reports/`）：集成测试报告与导出的文档。

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@misc{zhao2025sa-mil,
  title={Beyond Semantic Drift: High-Fidelity Byte-Level Malware Attribution via Structure-Aware MIL},
  author={Zhao, Wenzhou},
  year={2026}
}
```

## 许可证

本项目基于 MIT 许可证发布 — 参见 [LICENSE](LICENSE)。
