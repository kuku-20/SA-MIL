import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

from sklearn.metrics import confusion_matrix


def plot_training_curves(epochs, train_vals, val_vals, save_path='training_curve.png'):
    """绘制训练/验证曲线并保存到文件。"""
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_vals, marker='o', label='Train F1')
    plt.plot(epochs, val_vals, marker='o', label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=200)
        print(f"[generate_pictures] saved training curves to {save_path}")
    except Exception as e:
        print("[generate_pictures] failed to save training curves:", e)
    finally:
        plt.close()


def plot_confusion_matrix(labels, preds, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵并保存到文件。

    labels: list of true labels
    preds:  list of predicted labels
    class_names: list of class name strings (index order)
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.5)))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', cbar_kws={'label': 'Count'})
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar(label='Count')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.yticks(range(len(class_names)), class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=200)
        print(f"[generate_pictures] saved confusion matrix to {save_path}")
    except Exception as e:
        print("[generate_pictures] failed to save confusion matrix:", e)
    finally:
        plt.close()
