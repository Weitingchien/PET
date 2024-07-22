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

    pattern = r'BERT_self_training_(0\.\d(?:_balance_label)?) - (\w+)_pred\.txt:\nTotal Samples: \d+\nAccuracy: ([\d.]+)%\nMacro F1 Score: ([\d.]+)%'
    matches = re.findall(pattern, content, re.MULTILINE)

    data = {}
    for match in matches:
        model, dataset, accuracy, f1 = match
        if model not in data:
            data[model] = {}
        data[model][dataset] = {'Accuracy': float(accuracy), 'Macro F1': float(f1)}

    print(f"Parsed data: {data}")
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


def create_plots(data):
    models = sorted(data.keys())
    datasets = ['dev_v0', 'dev', 'test']
    metrics = ['Accuracy', 'Macro F1']
    
    for metric in metrics:
        plt.figure(figsize=(20, 12))
        
        for model in models:
            values = [data[model].get(dataset, {}).get(metric, 0) for dataset in datasets]
            plt.plot(datasets, values, marker='o', linewidth=3, markersize=12, label=f'{model}')

        plt.xlabel('Datasets', fontweight='bold', fontsize=24)
        plt.ylabel(f'{metric} (%)', fontweight='bold', fontsize=24)
        plt.title(f'Comparing {metric} at different ratios', fontweight='bold', fontsize=28)
        plt.legend(fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        for model in models:
            values = [data[model].get(dataset, {}).get(metric, 0) for dataset in datasets]
            for i, value in enumerate(values):
                plt.text(i, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=16)

        plt.tight_layout()
        plt.savefig(f'{metric}_comparison_line.png')
        plt.close()




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
    datasets_per_figure = 4  # Number of datasets to show in each figure

    for i in range(0, len(datasets), datasets_per_figure):
        current_datasets = datasets[i:i+datasets_per_figure]
        
        fig, axs = plt.subplots(len(current_datasets), 1, figsize=(20, 8*len(current_datasets)), squeeze=False)
        fig.suptitle(f'Label Counts for Datasets (Part {i//datasets_per_figure + 1})', fontsize=28, y=1.02)
        
        plt.subplots_adjust(hspace=0.6)  # 增加子圖之間的垂直間距

        for j, dataset in enumerate(current_datasets):
            if dataset in data:
                labels = list(data[dataset].keys())
                counts = list(data[dataset].values())
                
                axs[j, 0].bar(labels, counts)
                axs[j, 0].set_title(f'{dataset} Dataset', fontsize=26, pad=20)
                axs[j, 0].set_ylabel('Count', fontsize=24)
                axs[j, 0].tick_params(axis='both', which='major', labelsize=24)
                axs[j, 0].tick_params(axis='x')
                
                for k, count in enumerate(counts):
                    axs[j, 0].text(k, count, str(count), ha='center', va='bottom', fontsize=20)

        plt.tight_layout()
        plt.savefig(f'label_counts_part{i//datasets_per_figure + 1}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
        plt.close()




def main():
    file_path = './prediction_results.txt'  # 替換為你的文件路徑
    data = parse_file(file_path)
    print(f'data: {data}')
    create_plots(data)


    label_counts_file_path = './label_counts.txt'
    label_counts_data = parse_label_counts(label_counts_file_path)
    create_label_count_plot(label_counts_data)

if __name__ == "__main__":
    main()