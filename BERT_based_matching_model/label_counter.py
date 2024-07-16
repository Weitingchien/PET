# label_counter.py
from collections import defaultdict

def count_label_samples(file_path):
    label_count = defaultdict(int)
    total_labels = 0  # 初始化標籤總數計數器

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 忽略空行
                label = line.split('\t')[0]
                label_count[label] += 1
                total_labels += 1  # 每次遇到一個標籤就增加總數

    num_unique_labels = len(label_count)  # 計算不同標籤的數量
    print(f'({file_path})num_unique_labels: {num_unique_labels}')
    print(f'total_labels: {total_labels}')

                

    return label_count