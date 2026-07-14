
import matplotlib.pyplot as plt
import numpy as np

# Set academic plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

# ---------------------------------------------------------
# Data Preparation (Extracted from console log)
# ---------------------------------------------------------
test_f1_full = [0.8802, 0.9251, 0.9649, 0.9695, 0.9738, 0.9686, 0.9813, 0.9837, 0.9866, 0.9794, 
                0.9811, 0.9803, 0.9868, 0.9824, 0.9890, 0.9882, 0.9916, 0.9886, 0.9900, 0.9867, 
                0.9899,  0.9915, 0.9915, 0.9915, 0.9928]

test_f1_noflip = [0.8917, 0.9508, 0.9566, 0.9641, 0.9655, 0.9724, 0.9788, 0.9749, 0.9851, 0.9811, 
                  0.9789, 0.9751, 0.9810, 0.9828, 0.9831, 0.9808, 0.9828, 0.9825, 0.9867, 0.9825, 
                   0.9841,  0.9856, 0.9873, 0.9873, 0.9858]

test_f1_norand = [0.9081, 0.9637, 0.9683, 0.9817, 0.9754, 0.9836, 0.9802, 0.9775, 0.9767, 0.9792, 
                  0.9866, 0.9821, 0.9856, 0.9806, 0.9875, 0.9796, 0.9850, 0.9818, 0.9850, 0.9849, 
                  0.9812, 0.9822, 0.9836, 0.9865, 0.9836]

test_f1_none = [0.9043, 0.9489, 0.9606, 0.9711, 0.9801, 0.9762, 0.9789, 0.9834, 0.9870, 0.9879, 
                0.9859, 0.9865, 0.9828, 0.9810, 0.9843, 0.9830, 0.9842, 0.9841, 0.9883, 0.9873, 
                0.9820, 0.9859, 0.9842, 0.9842, 0.9875]

epochs_full = list(range(1, len(test_f1_full) + 1))
epochs_noflip = list(range(1, len(test_f1_noflip) + 1))
epochs_norand = list(range(1, len(test_f1_norand) + 1))
epochs_none = list(range(1, len(test_f1_none) + 1))

modes = ['None\n(Zero Aug)', 'No_Flip', 'No_RandAug', 'Full\n(Baseline)']
final_f1 = [0.9883, 0.9873, 0.9875, 0.9928]

# 🌸 Macaron Palette (Pastel Mint, Lavender, Peach, Coral Pink)
colors = ['#88d8b0', '#c3aed6', '#ffd3b6', '#ff8b94'] 
markers = ['D', 's', '^', 'o']

# ---------------------------------------------------------
# Plotting setup (1x2 Subplots)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2, 1.2]})

# --- Left Panel: Epoch vs F1 Curve ---
# 对于折线图，马卡龙色可能较浅，因此我们加上黑色边缘 (markeredgecolor) 确保高辨识度
ax1.plot(epochs_none, test_f1_none, marker=markers[0], markersize=6, color=colors[0], markeredgecolor='black', markeredgewidth=0.8, linewidth=2.5, label='None (No Augmentation)')
ax1.plot(epochs_noflip, test_f1_noflip, marker=markers[1], markersize=6, color=colors[1], markeredgecolor='black', markeredgewidth=0.8, linewidth=1.8, label='No_Flip', alpha=0.9)
ax1.plot(epochs_norand, test_f1_norand, marker=markers[2], markersize=6, color=colors[2], markeredgecolor='black', markeredgewidth=0.8, linewidth=1.8, label='No_RandAug', alpha=0.9)
ax1.plot(epochs_full, test_f1_full, marker=markers[3], markersize=6, color=colors[3], markeredgecolor='black', markeredgewidth=0.8, linewidth=1.8, label='Full (Baseline)', alpha=0.9)

ax1.set_xlabel('Training Epochs', fontweight='bold')
ax1.set_ylabel('Test Macro-F1', fontweight='bold')
ax1.set_title('(a) Robustness Over Epochs', fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_ylim(0.87, 1.00)
ax1.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)

# --- Right Panel: Final F1 Bar Chart ---
# 柱状图配合黑色边框在马卡龙配色下效果极佳
bars = ax2.bar(modes, final_f1, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

ax2.set_xlabel('Augmentation Mode', fontweight='bold')
ax2.set_ylabel('Best Macro-F1', fontweight='bold')
ax2.set_title('(b) Impact of Augmentations', fontweight='bold', pad=15)
ax2.set_ylim(0.97, 1.00) 
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# Add text annotations on top of bars
for bar, f1_val in zip(bars, final_f1):
    height = bar.get_height()
    weight = 'bold' if f1_val == 0.9883 else 'normal'
    ax2.annotate(f'{f1_val:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 6),  # 6 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=12, fontweight=weight)

# Adjust layout and save
plt.tight_layout()
plt.savefig('augmentation_ablation_macaron.pdf', format='pdf', dpi=300)
plt.savefig('augmentation_ablation_macaron.png', format='png', dpi=300)
print("Charts saved successfully as 'augmentation_ablation_macaron.pdf' and 'augmentation_ablation_macaron.png'.")
plt.show()
