import os
import csv


RAW_LABEL_CSV = "MalwareBazaar_Labels.csv"
OUTPUT_CSV = "Existing_labels.csv"
EXTRACTED_DIR = "malware_samples_pe"


def generate_actual_label_file(label_csv, output_csv, extracted_root):
    """根据解压目录中实际存在的 SHA256 文件生成新的标签 CSV。"""
    labels = {}
    with open(label_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = row['Malware SHA-256'].lower()
            labels[sha] = row

    present = set()
    for family in os.listdir(extracted_root):
        fam_dir = os.path.join(extracted_root, family)
        if not os.path.isdir(fam_dir):
            continue
        for fname in os.listdir(fam_dir):
            base, _ = os.path.splitext(fname)
            present.add(base.lower())

    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['Malware SHA-256', 'Family', 'Label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        matched = 0
        for sha in sorted(present):
            if sha in labels:
                writer.writerow(labels[sha])
                matched += 1
            else:
                writer.writerow({'Malware SHA-256': sha, 'Family': '', 'Label': ''})

    print(f"[*] 输出标签文件 {output_csv}, 样本数 {len(present)}, 已有标签 {matched}")


if __name__ == '__main__':
    generate_actual_label_file(RAW_LABEL_CSV, OUTPUT_CSV, EXTRACTED_DIR)
