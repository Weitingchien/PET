import numpy as np
from collections import Counter

def macro_f1(y_true, y_pred):
    """
    計算macro F1分數
    
    參數:
    y_true: 真實標籤列表
    y_pred: 預測標籤列表
    
    返回:
    macro_f1: macro F1分數
    """
    # 獲取所有唯一的類別
    labels = set(y_true + y_pred)
    
    f1_scores = []
    for label in labels:
        tp = sum((t == label) and (p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label) and (p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label) and (p != label) for t, p in zip(y_true, y_pred))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

# 示例使用
if __name__ == "__main__":
    # 假設的真實標籤和預測標籤
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    
    # 計算macro F1分數
    score = macro_f1(y_true, y_pred)
    print(f"Macro F1 score: {score:.4f}")