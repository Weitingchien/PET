import os
from collections import defaultdict

def calculate_correct_predictions(file_path):
    true_positives = 0  # 正確預測的正樣本數(TP)
    false_positives = 0  # 錯誤預測為正的樣本數(FP)
    class_tp = defaultdict(int)  # 每個類別的 True Positives
    class_total = defaultdict(int)  # 每個類別的總樣本數
    total_samples = 0
    sample_count = 0
    sample_id = 0
    current_true_label = None
    current_predicted_label = None
    is_true_label_one = False
    correct_samples = []
    correct_samples_id = []
    current_sample_number = 0

    with open(file_path, 'r') as file:
        print(f'file_path {file_path}')
        for line in file:
            if line.startswith("Sample "):
                sample_id = int(line.split()[1].strip(":")) - 1
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
                        correct_samples_id.append(sample_id)
                        correct_samples.append(current_sample_number)
                    else:
                         # 如果模型預測的標籤與這個樣本正確答案的標籤不一樣, FN+1
                        false_positives += 1
                """
                if is_true_label_one and current_true_label == current_predicted_label:
                    correct_samples_id.append(sample_id)
                    true_positives += 1
                    correct_samples.append(current_sample_number)
                """
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

    """
    # 處理最後一組樣本(如果不足10個)
    if sample_count > 0:
        total_samples += 1
        if is_true_label_one and current_true_label == current_predicted_label:
            correct_count += 1
            correct_samples.append(current_sample_number)
    """

    return true_positives, total_samples, correct_samples, correct_samples_id, class_tp, class_total

def calculate_macro_f1(class_tp, class_total):
    f1_scores = []
    for label in class_total:
        tp = class_tp[label]
        total = class_total[label]
        precision = tp / total if total > 0 else 0
        recall = precision  # 在這種情況下，Precision 和 Recall 相同
        f1 = precision  # F1 也與 Precision 和 Recall 相同
        f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return macro_f1





def process_folders():
    """
        TP(真陽性)= correct_count (正確預測的樣本數)
        FP(假陽性)= 模型預測為正但實際為負的樣本數
    """
    results = []
    for folder in ['BERT_self_training_0.2', 'BERT_self_training_0.2_balance_label', 'BERT_self_training_0.4', 'BERT_self_training_0.8']:
        for file in os.listdir(folder):
            if file.endswith('_pred.txt'):
                file_path = os.path.join(folder, file)
                true_positives, total_samples, correct_samples, correct_samples_id, class_tp, class_total = calculate_correct_predictions(file_path)
                macro_f1 = calculate_macro_f1(class_tp, class_total)
                total_tp = sum(class_tp.values())
                accuracy = total_tp / total_samples if total_samples > 0 else 0

                results.append(f"{folder} - {file}:")
                results.append(f"Total Samples: {total_samples}, Accuracy: {accuracy:.2%}")
                results.append(f"Macro F1 Score: {macro_f1:.2%}")
                results.append("Class-wise performance:")
                for label in class_total:
                    tp = class_tp[label]
                    total = class_total[label]
                    precision = tp / total if total > 0 else 0
                    results.append(f"  {label}: TP={tp}, Total={total}, Precision={precision:.2%}")
                results.append("")  # 空行分隔不同文件的結果
                # results.append(f"{folder} - {file}: Correct: {true_positives}, Total: {total}, Precision: {Precision:.2%}")
                # results.append(f"Correct samples: {correct_samples}")
                # results.append(f'len(samples):  {len(correct_samples_id)}')
                # results.append(f'samples: {correct_samples_id}')
    
    with open('prediction_results.txt', 'w') as output_file:
        for result in results:
            output_file.write(result + '\n')

def main():
    process_folders()

if __name__ == "__main__":
    main()