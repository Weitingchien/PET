import csv
from collections import defaultdict
import math


def main():
    # 讀取CSV文件
    with open('results.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        data = list(csv_reader)
        print(f'data: {data}')

    # 將相同pattern的測試集準確率提取出來
    pattern_accuracies = defaultdict(list)
    for row in data:
        output_folder = row['output_folder']
        #output_folder =  output_folder.split('/')[3]
        # print(f'output_folder: { output_folder}')
        # return
        pattern = output_folder.split('/')[3].split('-')[0]  # 提取pattern
        
        accuracy = float(row['test_set_after_training'])
        pattern_accuracies[pattern].append(accuracy)

    # 計算每個pattern的標準差
    for pattern, accuracies in pattern_accuracies.items():
        n = len(accuracies)
        mean = sum(accuracies) / n
        squared_diffs = [(x - mean) ** 2 for x in accuracies]
        variance = sum(squared_diffs) / (n - 1)
        std_dev = math.sqrt(variance)
        print(f"Pattern {pattern}: Standard Deviation = {std_dev:.5f}")

if __name__ == "__main__":
    main()