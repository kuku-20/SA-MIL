import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.models import mil_v11 as mil
from src.models.mil_v11 import MalwareDataset, collate_fn, evaluate_ensemble_tta
from src.training.visualize import plot_confusion_matrix


def main():
    base = Path(__file__).resolve().parent.parent.parent
    data_root = base / 'data' / 'malware_images'
    classified_root = base / 'src' / 'models'

    train_csv = data_root / 'train.csv'
    test_csv = data_root / 'test.csv'

    families = set()
    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            families.add(row['family'])
    families = sorted(families)

    class_to_idx = {fam: i for i, fam in enumerate(families)}
    idx_to_class = {i: fam for fam, i in class_to_idx.items()}

    # keep the training module globals consistent for evaluate_ensemble_tta()
    mil.CLASS_TO_IDX = class_to_idx
    mil.IDX_TO_CLASS = idx_to_class
    mil.NUM_CLASSES = len(families)

    test_ds = MalwareDataset(str(test_csv), str(data_root), class_to_idx, train=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model_paths = [
        str(base / 'results' / 'checkpoints' / 'best_mil_v11_seed42.pth'),
        str(base / 'results' / 'checkpoints' / 'best_mil_v11_seed123.pth'),
        str(base / 'results' / 'checkpoints' / 'best_mil_v11_seed456.pth'),
    ]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    acc, f1, preds, labels = evaluate_ensemble_tta(model_paths, test_loader, device)

    print(f'Final Test Accuracy: {acc:.4f}')
    print(f'Final Test Macro F1: {f1:.4f}')

    class_names = [idx_to_class[i] for i in range(len(families))]
    report = classification_report(labels, preds, target_names=class_names)
    print('\n' + report)

    report_path = base / 'results' / 'reports' / 'test_report_ensemble.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Final Test Accuracy: {acc:.4f}\n')
        f.write(f'Final Test Macro F1: {f1:.4f}\n\n')
        f.write(report)

    cm_path = base / 'results' / 'figures' / 'confusion_matrix_6class_ensemble.png'
    plot_confusion_matrix(labels, preds, class_names, save_path=str(cm_path))

    print(f'[*] 已保存: {report_path}')
    print(f'[*] 已保存: {cm_path}')


if __name__ == '__main__':
    main()
