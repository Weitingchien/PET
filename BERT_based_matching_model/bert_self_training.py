import os
import csv
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm
from itertools import count
from collections import Counter
from sklearn.metrics import f1_score

from label_counter import count_label_samples



def get_related_ids(sample_id, num_hypotheses):
    base_id = (sample_id - 1) * num_hypotheses + 1
    return set(range(base_id, base_id + num_hypotheses))



def process_prediction_output(file_path, class_to_hypothesis):
    samples = []
    current_sample = None
    sample_count = 0
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.startswith("Sample "):
                    sample_count += 1
                    if sample_count % len(class_to_hypothesis) == 1:
                        if current_sample:
                            samples.append(current_sample)
                        current_sample = {"probabilities": {}, "id": sample_count, "original_text": ""}  # 添加 original_text 字段
                elif line.startswith("Text: "):
                    current_sample["text"] = line.split("Text: ")[1].strip().split("[SEP]")[0].strip()
                elif line.startswith("Original Text: "):
                    current_sample["original_text"] = line.split("Original Text: ")[1].strip()
                elif line.startswith("True Label: "):
                    current_sample["true_label"] = line.split("True Label: ")[1].strip()
                elif line.startswith("Predicted Label: "):
                    current_sample["predicted_label"] = line.split("Predicted Label: ")[1].strip()
                elif ":" in line and not line.startswith("Probabilities:"):
                    label, prob = line.strip().split(":")
                    try:
                        current_sample["probabilities"][label.strip()] = float(prob.strip())
                    except ValueError:
                        print(f"Error converting probability to float on line {line_num}: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_num}: {line.strip()}")
                print(f"Error message: {str(e)}")

    if current_sample:
        samples.append(current_sample)
    
    # 计算每个样本的最高置信度
    for sample in samples:
        if sample["probabilities"]:
            sample["max_confidence"] = max(sample["probabilities"].values())
            sample["predicted_label"] = max(sample["probabilities"], key=sample["probabilities"].get)
    
    return samples

def parse_args():
    parser = argparse.ArgumentParser(description="BERT Self-training for Zero-shot Text Classification")
    parser.add_argument('--full_dataset', action='store_true', help='Use full dataset instead of 10 samples')
    parser.add_argument('--ratio', type=float, default=0.2, help='Selection ratio for self-training')
    parser.add_argument('--balance_label', action='store_true', help='Balance training samples by label')
    return parser.parse_args()



def evaluate(model, D_s_dev, D_u_dev, class_to_hypothesis):
    model.eval()
    
    f1_s = f1_score([item[0] for item in D_s_dev], 
                    [max(zip(class_to_hypothesis.keys(), item[2]), key=lambda x: x[1])[0] for item in D_s_dev], 
                    average='macro')
    
    f1_u = f1_score([item[0] for item in D_u_dev], 
                    [max(zip(class_to_hypothesis.keys(), item[2]), key=lambda x: x[1])[0] for item in D_u_dev], 
                    average='macro')
    
    return f1_s, f1_u



def train(model, optimizer, criterion, train_dataloader, tokenizer, max_length, device, num_epochs=1):
    total_batches = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in progress_bar:
            input_ids, attention_masks, labels, true_labels, ids = prepare_data(batch, tokenizer, max_length)
            
            # 將數據移到GPU
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            # 訓練匹配模型
            logits = model(input_ids, attention_masks)
            # 預測結果與真實標籤之間的二元交叉熵損失
            loss = criterion(logits.squeeze(), labels)
            
            optimizer.zero_grad() # 清零梯度
            loss.backward() # 反向傳播
            optimizer.step() # 更新參數

            total_loss += loss.item() # 累加損失
            
            # 更新進度條
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'})

        avg_loss = total_loss / total_batches # 計算平均損失
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")





"""
file name: train_pu_half_v0.txt
self.labels: 20464
unique_labels['anger', 'fear', 'love', 'shame', 'sadness']

file name: train_pu_half_v1.txt
self.labels: 14203
unique_labels['disgust', 'joy', 'surprise', 'guilt']

file name: dev.txt
self.labels: 7700
unique_labels['anger', 'fear', 'disgust', 'love', 'shame', 'sadness', 'guilt', 'joy', 'surprise', 'noemo']
"""



class EmotionDataset(Dataset):
    def __init__(self, file_path, max_samples=None):
        self.labels = []
        self.texts = []
        self.text_to_id = {}  # 新增：保存文本到ID的映射
        self.file_name = file_path.split('/')[3]
        print(f'file name: {self.file_name}')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                label, _, text = line.strip().partition('\t')
                source, _, text = text.partition('\t')
                self.labels.append(label)
                self.texts.append(text)
                self.text_to_id[text] = i + 1  # 使用索引 + 1 作為 ID

        # print(f'self.labels: {len(self.labels)} {self.labels}')

        self.unique_labels = list(set(self.labels))
        print(f'unique_labels{self.unique_labels}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return label, text
    
    def get_id_for_text(self, text):
        return self.text_to_id.get(text)

    def get_balanced_data(self, max_samples_per_class=None):
        label_counts = Counter(self.labels)
        if max_samples_per_class is None:
            max_samples_per_class = min(label_counts.values())

        balanced_data = []
        for label in set(self.labels):
            samples = [(l, t) for l, t in zip(self.labels, self.texts) if l == label]
            balanced_data.extend(random.sample(samples, max_samples_per_class))

        balanced_labels, balanced_texts = zip(*balanced_data)
        
        print(f'Balanced dataset. Total samples: {len(balanced_labels)}')
        print(f'Label distribution: {Counter(balanced_labels)}')
        
        return list(balanced_labels), list(balanced_texts)








# 定義Dataset
class MatchingDataset(Dataset):
    def __init__(self, data, class_to_hypothesis, label_map):
        self.data = []
        self.class_to_hypothesis = class_to_hypothesis
        self.label_map = label_map
        self.original_texts = []

        
        for id, label, text in data:
            for hyp_label, hypothesis in class_to_hypothesis.items():
                self.data.append((id, text, hypothesis, 1 if label == hyp_label else 0, label))
            self.original_texts.append(text)  # 儲存原始文本
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]







# 定義BERT-based matching model
class BertMatchingModel(nn.Module):
    def __init__(self, max_length=256):
        super(BertMatchingModel, self).__init__() #初始化
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 1) # 線性分類器: 輸入維度為768, 輸出維度為1
        self.max_length = max_length # 序列的最大長度
    # 定義模型的前向傳播方法, 接受input_ids和attention_mask作為輸入
    def forward(self, input_ids, attention_mask):
        # 輸入傳遞給BERT模型, 獲取輸出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # outputs[0]: 最後一層的hidden state (形狀為 [batch_size, 256, 768])
        """ [:]: 選擇所有批次(保留 batch 維度)
            [0]: 選擇每個序列的第一個 token(即 [CLS] token)
            [:]: 選擇該 token 的所有特徵維度
        """
        # cls_output的形狀為 [batch_size, 768]
        cls_output = outputs[0][:, 0, :]  # 取出[CLS]對應的hidden state (c_(x,y'))
        # [CLS] token表示傳入線性分類器, 得到標量(logits),  W^T * c_(x,y') + b
        logits = self.classifier(cls_output)
        # sigmoid函數, 輸出壓縮至0-1之間, 返回最終的預測機率(二元分類任務)
        return torch.sigmoid(logits)

# 準備訓練數據
def prepare_data(batch, tokenizer, max_length=256):
    input_ids = []
    attention_masks = []
    labels = []
    true_labels = []  # 新增：保存真實標籤
    ids =  []
    
    for id, text, hypothesis, binary_label, true_label in batch:
        # 輸入格式：[CLS] text [SEP] hypothesis [SEP]
        encoded = tokenizer.encode_plus(
            text,
            hypothesis,
            add_special_tokens=True,
            max_length=max_length,
            padding=False,  # 不使用自動填充
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        
        # 手動填充
        seq_len = encoded['input_ids'].size(1)
        if seq_len < max_length:
            padding_length = max_length - seq_len
            # .squeeze()：移除大小為1的維度。例如，如果encoded['input_ids']的形狀是[1, seq_len]，它會變成[seq_len]
            # torch.zeros(padding_length, dtype=torch.long)：建立一個長度為padding_length的零張量
            # torch.zeros(padding_length, dtype=torch.long)], dim=0), 將兩個張量在維度0（即序列長度維度）上連接
            # 整體效果：如果原始序列長度小於max_length，這行代碼會在序列後面添加零，直到達到max_length
            input_id = torch.cat([encoded['input_ids'].squeeze(), torch.zeros(padding_length, dtype=torch.long)], dim=0)
            # 生成一個與輸入ID等長的遮罩，其中1表示實際內容，0表示填充部分
            # 在自注意力計算中，填充位置（0）不會被考慮，模型只關注實際內容(防止填充干擾)
            attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)], dim=0)
        else:
            input_id = encoded['input_ids'].squeeze()[:max_length]
            attention_mask = torch.ones(max_length, dtype=torch.long)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(binary_label)
        true_labels.append(true_label)  # 新增：保存真實標籤
        ids.append(id)
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return input_ids, attention_masks, labels, true_labels, ids


def predict(model, dataloader, tokenizer, max_length, device, class_to_hypothesis, output_file):
    model.eval()
    confidences = []
    original_texts = []
    
    with torch.no_grad(), open(output_file, 'w') as f:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            input_ids, attention_masks, labels, true_labels, ids = prepare_data(batch, tokenizer, max_length)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            outputs = model(input_ids, attention_masks)
            probabilities = outputs.squeeze().cpu().numpy()
            
            class_list = list(class_to_hypothesis.keys())
            class_prob_map = dict(zip(class_list, probabilities))
            # print(f'class_prob_map: {class_prob_map}')
            predicted_label = max(class_prob_map, key=class_prob_map.get)
            
            for i, ((id, text, hypothesis, _, true_label), prob) in enumerate(zip(batch, probabilities)):
                full_input_text = tokenizer.decode(input_ids[i].cpu().tolist(), skip_special_tokens=False)
                
                f.write(f"Sample {batch_idx * len(batch) + i}:\n")
                f.write(f"Text: {full_input_text}\n")
                f.write(f"Original Text: {text}\n")
                f.write(f"True Label: {true_label}\n")
                f.write(f"Predicted Label: {predicted_label}\n")
                f.write("Probabilities:\n")
                for cls, prob in class_prob_map.items():
                    f.write(f"  {cls}: {prob}\n")
                f.write("\n")
                
                confidences.append((id, text, prob))
                original_texts.append(text)
    
    # print(f'predict: {confidences}')
    return confidences, original_texts



def evaluate_and_print_results(model, class_to_hypothesis, D_s_dev, D_p):
    f1_s, _ = evaluate(model, D_s_dev, [], class_to_hypothesis)
    _, f1_u = evaluate(model, [], D_p, class_to_hypothesis)
    print(f"F1 score on seen classes: {f1_s:.4f}")
    print(f"F1 score on unseen classes: {f1_u:.4f}")




def create_label_map(train_dataset, class_to_hypothesis):
    all_labels = set([label for label, _ in train_dataset] + list(class_to_hypothesis.keys()))
    return {label: idx for idx, label in enumerate(sorted(all_labels))}



def calculate_selection_number(total_unlabeled_samples, current_iteration, selection_ratio=0.8):
    return int((selection_ratio / current_iteration) * total_unlabeled_samples)


def get_label_from_confidence(probabilities, class_to_hypothesis):
    class_list = list(class_to_hypothesis.keys())
    class_prob_map = dict(zip(class_list, probabilities))
    # print(f'class_prob_map: {class_prob_map}')
    predicted_label = max(class_prob_map, key=class_prob_map.get)
    return class_list[np.argmax(probabilities)]






def evaluate_model(model, dataloader, tokenizer, max_length, device, class_to_hypothesis, output_file):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad(), open(output_file, 'w') as f:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids, attention_masks, labels, true_labels, ids  = prepare_data(batch, tokenizer, max_length)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            outputs = model(input_ids, attention_masks)
            probabilities = outputs.cpu().numpy()


            # print(f'type(input_ids): {input_ids}')
            # print(f'probabilities: {probabilities} len(probabilities):{len(probabilities)}' )
            # print(f'labels: {true_labels}')


            # 獲取類別列表
            class_list = list(class_to_hypothesis.keys())
            class_prob_map = dict(zip(class_list, probabilities))
            # print(f'class_prob_map: {class_prob_map}')
            predicted_label = max(class_prob_map, key=class_prob_map.get)


            # 解碼文本
            for i, (id, text, hypothesis, true_label, true_labels) in enumerate(batch):
                full_input_text = tokenizer.decode(input_ids[i].cpu().tolist(), skip_special_tokens=False)
                # 將預測結果寫入文件
                f.write(f"Sample {batch_idx * len(batch) + i}:\n")
                f.write(f"Text: {full_input_text}\n")
                f.write(f"True Label: {true_label}\n")
                f.write(f"Predicted Label: {predicted_label}\n")
                f.write("Probabilities:\n")
                for cls, prob in class_prob_map.items():
                    f.write(f"  {cls}: {prob}\n")
                f.write("\n")

            all_predictions.append(predicted_label)
            all_labels.append(true_labels)

    
    # 確保所有標籤的類型一致（這裡選擇使用字符串類型）
    all_predictions = [str(label) for label in all_predictions]
    all_labels = [str(label) for label in all_labels]

    print(f'len(all_predictions): {len(all_predictions)}')
    print(f'len(all_labels): {len(all_labels)}')

    
    # 計算 F1 分數
    f1 = f1_score(all_labels, all_predictions, average='macro')
    return f1



def main():
    
    args = parse_args()
    max_samples = None if args.full_dataset else 100
    ratio = args.ratio
    balance_label = args.balance_label
    


    # 初始化記錄列表
    dev_f1_scores = []
    dev_v0_scores = []
    test_f1_scores = []

    


    # 數據集路徑
    train_file = './datasets/emotion/train_pu_half_v0.txt'
    unseen_data_file = './datasets/emotion/train_pu_half_v1.txt'
    dev_file = './datasets/emotion/dev.txt'
    dev_v0_file = './datasets/emotion/dev_v0.txt'
    dev_v1_file = './datasets/emotion/dev_v1.txt'
    test_file = './datasets/emotion/test.txt'
    test_file_I = './datasets/emotion/unseen_class_test_I.txt'
    """
    # 統計每個數據集的標籤樣本數(寫入至label_counts.txt檔案)
    dataset_files = [
        ('train', train_file),
        ('unseen', unseen_data_file),
        ('dev', dev_file),
        ('test', test_file),
        ('(unseen)test', test_file_I),
        ('dev_v0', dev_v0_file),
        ('dev_v1', dev_v1_file)
    ]

    # 將標籤統計結果寫入txt文件
    with open('label_counts.txt', 'w', encoding='utf-8') as f:
        for dataset_name, file_path in dataset_files:
            label_counts = count_label_samples(file_path)
            f.write(f"Dataset: {dataset_name}\n")
            for label, count in label_counts.items():
                f.write(f"{label}: {count} samples\n")
            f.write("\n")
    return
    """
    
    

    counter = count(start=1)
    train_dataset = EmotionDataset(train_file, max_samples=max_samples)
    if balance_label:
        folder_name = f"BERT_self_training_{ratio}_balance_label"
        balanced_labels, balanced_texts = train_dataset.get_balanced_data()
        train_data = [(next(counter), label, text) for label, text in zip(balanced_labels, balanced_texts)]
    else:
        folder_name = f"BERT_self_training_{ratio}"
        train_data = [(next(counter), label, text) for label, text in train_dataset]

    print(f'len(train_data): {len(train_data)}')
    os.makedirs(folder_name, exist_ok=True)
    

    unseen_dataset = EmotionDataset(unseen_data_file, max_samples=max_samples)
    dev_dataset = EmotionDataset(dev_file, max_samples=max_samples)
    dev_v0_dataset = EmotionDataset(dev_v0_file, max_samples=max_samples)
    test_dataset = EmotionDataset(test_file_I, max_samples=max_samples)
    unseen_dataset_with_id = [(next(counter), label, text) for label, text in unseen_dataset]

    # 假設有一下seen classes和對應的hypothesis(train_pu_half_v0.txt訓練集的標籤描述)
    # 已知類別（用於訓練）
    seen_class_to_hypothesis = {
        'shame': 'This text expresses shame.',
        'sadness': 'This text expresses sadness.',
        'love': 'This text expresses love.',
        'fear': 'This text expresses fear.',
        'anger': 'This text expresses anger.',
    }
    """
    unseen_class_to_hypothesis = {
        'joy': 'This text expresses joy.',
        'guilt': 'This text expresses guilt.',
        'disgust': 'This text expresses disgust.',
        'surprise': 'This text expresses surprise.',
    }
    """

    # 所有可能的類別（用於評估和預測）
    all_class_to_hypothesis = {
        'shame': 'This text expresses shame.',
        'sadness': 'This text expresses sadness.',
        'love': 'This text expresses love.',
        'fear': 'This text expresses fear.',
        'anger': 'This text expresses anger.',
        'disgust': 'This text expresses disgust.',
        'joy': 'This text expresses joy.',
        'surprise': 'This text expresses surprise.',
        'guilt': 'This text expresses guilt.',
        'noemo': 'This text expresses noemo.',
    }








    label_map = create_label_map(train_dataset, seen_class_to_hypothesis)
    print(f'label_map: {label_map}')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 找訓練集文本最大長度的那個文本, 當作最大文本長度
    max_text_len = max(len(tokenizer.encode(text)) for _, text in train_dataset)
    max_hyp_len = max(len(tokenizer.encode(hyp)) for hyp in all_class_to_hypothesis.values())
    print(f"Max text length: {max_text_len}, Max hypothesis length: {max_hyp_len}")
    max_length = min(512, max_text_len + max_hyp_len + 3)  # +3 for [CLS] and [SEP] tokens
    print(f"Using max_length: {max_length}")

    # 初始化model和tokenizer
    model = BertMatchingModel(max_length=max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # 損失函數
    criterion = nn.BCELoss()

    # 設定超參數
    num_iterations = 5
    early_stop_patience = 3

    # Initialize pseudo-labeled data D^p
    D_p = []
    best_f1 = 0
    no_improvement_count = 0

    unseen_data = unseen_dataset_with_id.copy()

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")
        # 訓練matching model f
        current_train_data = train_data + D_p
        

        # seen dataset (使用 seen_class_to_hypothesis 進行訓練)
        train_dataloader = DataLoader(MatchingDataset(current_train_data, seen_class_to_hypothesis, label_map), 
                                      batch_size=10, shuffle=True, collate_fn=lambda x: x, num_workers=12)
        # 開始訓練
        train(model, optimizer, criterion, train_dataloader, tokenizer, max_length, device)

        # unseen dataset(使用all_class_to_hypothesis 進行預測)
        unseen_dataloader = DataLoader(MatchingDataset(unseen_data, all_class_to_hypothesis, label_map), 
                                       batch_size=10, shuffle=False, collate_fn=lambda x: x, num_workers=12)
        # 在未標籤數據集上預測
        csv_path = os.path.join(folder_name, 'unseen_predict_output.txt')
        confidences, original_texts = predict(model, unseen_dataloader, tokenizer, max_length, device, all_class_to_hypothesis, csv_path)
        
        # 預測結果儲存至unseen_predict_output.txt, 並逐行讀取unseen_predict_output.txt
        processed_samples = process_prediction_output(csv_path, all_class_to_hypothesis)
        
        sorted_samples = sorted(processed_samples, key=lambda x: x["max_confidence"], reverse=True)

        # samples = [sample for sample in sorted_samples]
        # print(f'samples{samples}')
        
        # 計算應選擇的樣本數量
        initial_unlabeled_samples = len(unseen_data)
        # print(f'initial_unlabeled_samples: {initial_unlabeled_samples}')
        selection_number = calculate_selection_number(initial_unlabeled_samples, iteration + 1, ratio)
        selection_number_file = os.path.join(folder_name, 'selection_numbers.txt')
        with open(selection_number_file, 'a') as f:
            f.write(f"Iteration {iteration + 1}: {selection_number}\n")
        # print(f'selection_number: {selection_number}')

        # 選擇高置信度的樣本
        new_pseudo_labeled = [(unseen_dataset.get_id_for_text(sample["original_text"]), sample["original_text"], sample["predicted_label"]) 
                      for sample in sorted_samples[:selection_number]]
        # print(f'new_pseudo_labeled: {new_pseudo_labeled} len(new_pseudo_labeled): {len(new_pseudo_labeled)}')

        # 更新偽標籤數據集
        D_p.extend(new_pseudo_labeled)
        selected_ids = set()
        for id, _, _ in new_pseudo_labeled:
            # 找到當前樣本的其他正負樣本, 例如Selected IDs: {16, 2, 93}對於16這個ID就會有10至19的ID
            selected_ids.update(get_related_ids(id, len(all_class_to_hypothesis)))
        sorted_selected_ids = sorted(selected_ids)
        # print(f"Selected IDs: {sorted_selected_ids}")

        print(f"Unseen dataset size before removal: {len(unseen_data)}")
        # 從未標記數據集中移除被選中的樣本
        unseen_data = [item for item in unseen_data if item[0] not in sorted_selected_ids]
        print(f"Unseen dataset size after removal: {len(unseen_data)}")



        # 評估當前模型(評估時使用 all_class_to_hypothesis)
        dev_dataloader = DataLoader(MatchingDataset([(next(counter), label, text) for label, text in dev_dataset], 
                            all_class_to_hypothesis, label_map), 
                            batch_size=10, shuffle=False, collate_fn=lambda x: x, num_workers=12)

        dev_v0_dataloader = DataLoader(MatchingDataset([(next(counter), label, text) for label, text in dev_v0_dataset], 
                            all_class_to_hypothesis, label_map), 
                            batch_size=10, shuffle=False, collate_fn=lambda x: x, num_workers=12)

        csv_path = os.path.join(folder_name, 'dev_pred.txt')
        dev_f1_score = evaluate_model(model, dev_dataloader, tokenizer, max_length, device, all_class_to_hypothesis, csv_path)
        csv_path = os.path.join(folder_name, 'dev_v0_pred.txt')
        dev_v0_score = evaluate_model(model, dev_v0_dataloader, tokenizer, max_length, device, all_class_to_hypothesis, csv_path)
        print(f"Dev F1 score: {dev_f1_score:.4f}")
        print(f"Dev_v0 F1 score: {dev_v0_score:.4f}")

        # 記錄 Dev F1 score
        dev_f1_scores.append(dev_f1_score)
        dev_v0_scores.append(dev_v0_score)

        # Early stopping
        if dev_f1_score > best_f1:
            best_f1 = dev_f1_score
            no_improvement_count = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop_patience:
            print("Early stopping")
            break

    # 加載最佳模型並在測試集上評估
    model.load_state_dict(torch.load("best_model.pt"))

    # 最終測試時使用 all_class_to_hypothesis
    test_dataloader = DataLoader(MatchingDataset([(next(counter), label, text) for label, text in test_dataset], 
                             all_class_to_hypothesis, label_map), 
                             batch_size=10, shuffle=False, collate_fn=lambda x: x, num_workers=12)

    csv_path = os.path.join(folder_name, 'test_pred.txt')
    test_f1_score = evaluate_model(model, test_dataloader, tokenizer, max_length, device, all_class_to_hypothesis, csv_path)
    
    print(f"Test F1 score: {test_f1_score:.4f}")

    # 記錄 Test F1 score
    test_f1_scores.append(test_f1_score)
    


    
    # 建立資料夾
    # os.makedirs(folder_name, exist_ok=True)
    csv_path = os.path.join(folder_name, 'f1_scores.csv')
    # 寫入f1_scores.csv
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Dev F1", "Dev_v0 F1", "Test F1"])
        for i in range(len(dev_f1_scores)):
            row = [i+1, dev_f1_scores[i], dev_v0_scores[i]]
            if i < len(test_f1_scores):
                row.append(test_f1_scores[i])
            else:
                row.append('')  # 如果沒有對應的測試分數, 添加空字串
            writer.writerow(row)


    print("F1 scores have been saved to f1_scores.csv")

if __name__ == "__main__":
    main()