import os
from collections import defaultdict

def calculate_correct_predictions(file_path):
    true_positives = 0  # 正確預測的正樣本數(TP)
    class_tp = defaultdict(int)  # 每個類別的 True Positives
    class_fp = defaultdict(int)  # 每個類別的 False Positives
    class_fn = defaultdict(int)  # 每個類別的 False Negatives
    class_total = defaultdict(int)  # 每個類別的總樣本數
    total_samples = 0
    sample_count = 0
    current_true_label = None
    current_predicted_label = None
    is_true_label_one = False

    with open(file_path, 'r') as file:
        print(f'file_path {file_path}')
        for line in file:
            if line.startswith("Sample "):
                sample_count += 1
                if sample_count == 1:
                    # 第幾個樣本
                    current_sample_number = int(line.split()[1].strip(':'))
                #is_true_label_one  == 1代表這個樣本是正確答案
                if is_true_label_one:
                    class_total[current_true_label] += 1
                    # 如果模型預測的標籤與這個樣本正確答案的標籤一樣, TP+1
                    if current_true_label == current_predicted_label:
                        class_tp[current_true_label] += 1
                        true_positives += 1
                    else:
                        # 如果模型預測的標籤與這個樣本正確答案的標籤不一樣
                        class_fn[current_true_label] += 1
                        class_fp[current_predicted_label] += 1

                if sample_count == 10:
                    total_samples += 1
                    sample_count = 0
                    current_true_label = None
                    current_predicted_label = None
                    is_true_label_one = False
            elif "this text expresses" in line:
                current_true_label = line.split("this text expresses ")[1].split(".")[0].strip()
            elif line.startswith("True Label:"):
                is_true_label_one = line.split(": ")[1].strip() == "1"
            elif line.startswith("Predicted Label:"):
                current_predicted_label = line.split(": ")[1].strip()

    return true_positives, total_samples, class_tp, class_fp, class_fn, class_total





def calculate_f1_scores(class_tp, class_fp, class_fn):
    # 計算整體的 TP, FP, FN
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    # 計算整體的 Precision 和 Recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    # 計算整體的 F1 分數
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

 
    
    return f1






def calculate_macro_f1(class_tp, class_fp, class_fn):
    f1_scores = []
    for label in set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys())):
        tp = class_tp[label]
        fp = class_fp[label]
        fn = class_fn[label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > 0:  # 只有當 F1 分數大於 0 時才加入計算
            f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return macro_f1




def process_folders():
    results = []
    for folder in ['BERT_self_training_0.1', 'BERT_self_training_0.2', 'BERT_self_training_0.2_balance_label', 'BERT_self_training_0.4', 'BERT_self_training_0.4_balance_label', 'BERT_self_training_0.8', 'BERT_self_training_0.8_balance_label']:
        for file in os.listdir(folder):
            if file.endswith('_pred.txt'):
                file_path = os.path.join(folder, file)
                true_positives, total_samples, class_tp, class_fp, class_fn, class_total = calculate_correct_predictions(file_path)

                # 計算準確率
                accuracy = true_positives / total_samples if total_samples > 0 else 0
                
                # 計算macro F1
                macro_f1 = calculate_macro_f1(class_tp, class_fp, class_fn)
                """
                # 計算整體 F1 和修改後的 Macro F1
                overall_f1 = calculate_f1_scores(class_tp, class_fp, class_fn)
                """
                
                
                results.append(f"{folder} - {file}:")
                results.append(f"Total Samples: {total_samples}")
                results.append(f"Accuracy: {accuracy:.2%}")
                # results.append(f"Overall F1 Score: {overall_f1:.2%}")
                results.append(f"Macro F1 Score: {macro_f1:.2%}")
                results.append("Class-wise performance:")
                for label in set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys())):
                    tp = class_tp[label]
                    fp = class_fp[label]
                    fn = class_fn[label]
                    total = class_total[label]
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    results.append(f"  {label}: TP={tp}, FP={fp}, FN={fn}, Total={total}, Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}")
                results.append("")  # 空行分隔不同文件的结果
    
    with open('prediction_results.txt', 'w') as output_file:
        for result in results:
            output_file.write(result + '\n')

def main():
    process_folders()

if __name__ == "__main__":
    main()