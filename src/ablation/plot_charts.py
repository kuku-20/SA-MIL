
import matplotlib.pyplot as plt
import numpy as np

# 数据准备
labels = ['Baseline [R]\n(Raw Byte)', 'Byte + Structure [RG]', 'Byte + Entropy [RB]', 'Proposed [RGB]\n(Synergy)']
acc_scores = [93.00, 95.17, 95.33, 98.67]
f1_scores = [93.17, 95.33, 95.41, 98.78]

x = np.arange(len(labels))
width = 0.35  

# 设置高颜值的学术配色
color_acc = '#4C72B0'  # 莫兰迪蓝
color_f1 = '#DD8452'   # 莫兰迪橙

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, acc_scores, width, label='Test Accuracy (%)', color=color_acc, edgecolor='black', zorder=3)
rects2 = ax.bar(x + width/2, f1_scores, width, label='Test Macro F1 (%)', color=color_f1, edgecolor='black', zorder=3)

# 坐标轴与标签
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
ax.set_title('Channel Ablation Study Results', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylim(90, 100) # 将Y轴起点设为90，放大差距视觉效果

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 添加图例
ax.legend(loc='upper left', fontsize=12)

# 在柱子上添加具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 向上偏移 3 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
save_path = "channel_ablation_chart.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存为 {save_path}")
