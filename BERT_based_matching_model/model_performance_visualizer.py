import re
import matplotlib.pyplot as plt
import numpy as np

"""
def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正則表達式解析數據
    pattern = r'BERT_self_training_(0\.\d) - (\w+)_pred\.txt:\nTotal Samples: \d+, Accuracy: ([\d.]+)%\nMacro F1 Score: ([\d.]+)%'
    matches = re.findall(pattern, content)

    data = {}
    for match in matches:
        model, dataset, accuracy, f1 = match
        if model not in data:
            data[model] = {}
        data[model][dataset] = {'Accuracy': float(accuracy), 'Macro F1': float(f1)}

    return data

"""

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 修改正則表達式只匹配 Accuracy
    pattern = r'BERT_self_training_(0\.\d(?:_balance_label)?) - (\w+)_pred\.txt:\nTotal Samples: \d+, Accuracy: ([\d.]+)%'
    matches = re.findall(pattern, content)

    data = {}
    for match in matches:
        model, dataset, accuracy = match
        if model not in data:
            data[model] = {}
        data[model][dataset] = {'Accuracy': float(accuracy)}

    return data






"""
def create_plot(data):
    models = sorted(data.keys())
    datasets = ['dev_v0', 'dev', 'test']
    metrics = ['Accuracy', 'Macro F1']

    # 設置圖形大小
    plt.figure(figsize=(15, 10))

    # 設置條形寬度
    bar_width = 0.1
    index = np.arange(len(datasets))

    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            values = [data[model].get(dataset, {}).get(metric, 0) for dataset in datasets]
            offset = bar_width * (i * len(metrics) + j)
            plt.bar(index + offset, values, bar_width, 
                    label=f'{metric} {model}')

    # 添加標籤和標題
    plt.xlabel('Datasets', fontweight='bold')
    plt.ylabel('Scores (%)', fontweight='bold')
    plt.title('Model Performance Comparison', fontweight='bold')
    plt.xticks(index + bar_width * 2.5, datasets)

    # 添加圖例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 顯示數值
    for rect in plt.gca().patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 顯示圖形
    plt.tight_layout()
    plt.show()
"""


def create_plot(data):
    models = sorted(data.keys())
    datasets = ['dev_v0', 'dev', 'test']
    
    # 設置圖形大小
    plt.figure(figsize=(15, 10))

    # 設置條形寬度
    bar_width = 0.2
    index = np.arange(len(datasets))

    for i, model in enumerate(models):
        values = [data[model].get(dataset, {}).get('Accuracy', 0) for dataset in datasets]
        offset = bar_width * i
        plt.bar(index + offset, values, bar_width, label=f'Accuracy (Ratio {model})')

    # 添加標籤和標題
    plt.xlabel('Datasets', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Comparing the accuracy at different ratios', fontweight='bold')
    plt.xticks(index + bar_width, datasets)

    # 添加圖例
    plt.legend()

    # 顯示數值
    for rect in plt.gca().patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 顯示圖形
    plt.tight_layout()
    plt.show()




def parse_label_counts(file_path):
    data = {}
    current_dataset = ""
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Dataset:"):
                current_dataset = line.split(":")[1].strip()
                data[current_dataset] = {}
            elif ":" in line:
                label, count = line.split(":")
                data[current_dataset][label.strip()] = int(count.split()[0])
    return data

def create_label_count_plot(data):
    datasets = ['train', 'unseen', 'dev', 'test', '(unseen)test', 'dev_v0', 'dev_v1']
    
    fig, axs = plt.subplots(len(datasets), 1, figsize=(15, 5*len(datasets)), squeeze=False)
    fig.suptitle('Label Counts for Each Dataset', fontsize=16)

    for i, dataset in enumerate(datasets):
        if dataset in data:
            labels = list(data[dataset].keys())
            counts = list(data[dataset].values())
            
            axs[i, 0].bar(labels, counts)
            axs[i, 0].set_title(f'{dataset} Dataset')
            axs[i, 0].set_ylabel('Count')
            axs[i, 0].tick_params(axis='x', rotation=45)
            
            for j, count in enumerate(counts):
                axs[i, 0].text(j, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()





def main():
    file_path = './prediction_results.txt'  # 替換為你的文件路徑
    data = parse_file(file_path)
    create_plot(data)


    label_counts_file_path = './label_counts.txt'
    label_counts_data = parse_label_counts(label_counts_file_path)
    create_label_count_plot(label_counts_data)

if __name__ == "__main__":
    main()