from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer, f1_score

import numpy as np

def ordered_f1_score(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    precision = correct / len(y_pred) if y_pred else 0
    recall = correct / len(y_true) if y_true else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1



def main():
    import torch
    import time
    print(torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("GPU型號:", torch.cuda.get_device_name(0))

        # 在CPU上創建一個大矩陣
        cpu_a = torch.randn(10000, 10000)
        cpu_b = torch.randn(10000, 10000)

        # 在GPU上創建相同的矩陣
        gpu_a = cpu_a.cuda()
        gpu_b = cpu_b.cuda()

        # 在CPU上進行矩陣乘法
        start = time.time()
        cpu_result = torch.matmul(cpu_a, cpu_b)
        cpu_time = time.time() - start
        print(f"CPU計算時間: {cpu_time:.4f} 秒")

        # 在GPU上進行矩陣乘法
        start = time.time()
        gpu_result = torch.matmul(gpu_a, gpu_b)
        torch.cuda.synchronize()  # 確保GPU運算完成
        gpu_time = time.time() - start
        print(f"GPU計算時間: {gpu_time:.4f} 秒")

        # 計算加速比
        speedup = cpu_time / gpu_time
        print(f"GPU加速比: {speedup:.2f}x")
    else:
        print("CUDA不可用, 無法進行GPU測試")
    return

    all_predictions = ['fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear']
    all_labels = ['fear', 'fear', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear', 'joy', 'fear'] 
    print(ordered_f1_score(all_labels, all_predictions))

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'f1: {f1}')


    y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 score: {macro_f1}")

if __name__ == "__main__":
    main()