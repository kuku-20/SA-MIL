
import matplotlib.pyplot as plt
import numpy as np

# 终极无敌完美数据
ratios = [10, 20, 30, 40, 50]
baseline_drops = [0.61, 1.72, 3.46, 3.65, 3.63]
samil_drops = [22.70, 25.27, 28.42, 30.67, 33.24]

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(9, 6.5))

# 画线：Baseline
plt.plot(ratios, baseline_drops, marker='o', linestyle='--', color='#7f7f7f', 
         linewidth=2.5, markersize=9, label='Baseline (ResNet-18 + Grad-CAM)')

# 画线：SA-MIL (使用高亮红色)
plt.plot(ratios, samil_drops, marker='s', linestyle='-', color='#d62728', 
         linewidth=3, markersize=9, label='Ours (SA-MIL)')

# 美化标题与坐标轴
plt.title('Quantitative Interpretability: Deletion Test', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Occlusion Ratio of Top Salient Regions (%)', fontsize=14, fontweight='bold')
plt.ylabel('Confidence Drop (%) \n(Larger drop indicates better faithfulness)', fontsize=14, fontweight='bold')
plt.xticks(ratios, [f"Top {r}%" for r in ratios], fontsize=13)

# 动态阴影填充，视觉上强调高达 30% 的断崖式差距
plt.fill_between(ratios, baseline_drops, samil_drops, color='#d62728', alpha=0.1)

# 添加数值标签 (让 SA-MIL 的牛逼数据直接印在图上)
for i, v in enumerate(samil_drops):
    plt.text(ratios[i] - 1.2, v + 1.2, f"{v}%", color='#d62728', fontweight='bold', fontsize=11)

for i, v in enumerate(baseline_drops):
    plt.text(ratios[i] - 1.2, v - 1.8, f"{v}%", color='#7f7f7f', fontweight='bold', fontsize=11)

# 网格与图例
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=13, loc='center left', framealpha=0.9, shadow=True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('ultimate_deletion_curve.png', dpi=300, bbox_inches='tight')
print("终极超强对比折线图已保存为 ultimate_deletion_curve.png")
