import torch
import random
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm
from sklearn.metrics import f1_score

def evaluate(model, D_s_dev, D_u_dev, class_to_hypothesis):
    model.eval()
    
    f1_s = f1_score([item[0] for item in D_s_dev], 
                    [max(zip(class_to_hypothesis.keys(), item[2]), key=lambda x: x[1])[0] for item in D_s_dev], 
                    average='weighted')
    
    f1_u = f1_score([item[0] for item in D_u_dev], 
                    [max(zip(class_to_hypothesis.keys(), item[2]), key=lambda x: x[1])[0] for item in D_u_dev], 
                    average='weighted')
    
    return f1_s, f1_u



def train(model, optimizer, criterion, train_dataloader, tokenizer, max_length, device, num_epochs=1):
    total_batches = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in progress_bar:
            input_ids, attention_masks, labels = prepare_data(batch, tokenizer, max_length)
            
            # 將數據移到GPU
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            # 訓練匹配模型
            logits = model(input_ids, attention_masks)
            
            loss = criterion(logits.squeeze(), labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # 更新進度條
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'})

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")





def update_policy_network(policy_network, optimizer_policy, states, actions, rewards):
    log_probs = []
    for state, action in zip(states, actions):
        action_probs = policy_network(state)
        log_prob = torch.log(action_probs[action])
        log_probs.append(log_prob)

    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    loss = -torch.mean(log_probs * rewards)
    
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()



class EmotionDataset_list(Dataset):
    def __init__(self, data):
        self.labels = []
        self.texts = []
        
        for label, text in data:
            self.labels.append(label)
            self.texts.append(text)

        self.unique_labels = list(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return label, text

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
    def __init__(self, file_path):
        self.labels = []
        self.texts = []
        self.file_name = file_path.split('/')[3]
        print(f'file name: {self.file_name}')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, _, text = line.strip().partition('\t')
                source, _, text = text.partition('\t')
                self.labels.append(label)
                self.texts.append(text)

        print(f'self.labels: {len(self.labels)}')

        self.unique_labels = list(set(self.labels))
        print(f'unique_labels{self.unique_labels}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        return label, text



def compute_rewards(f_s_list, f_u_list, lambda_value):
    mu_s, sigma_s = np.mean(f_s_list), np.std(f_s_list)
    mu_u, sigma_u = np.mean(f_u_list), np.std(f_u_list)
    
    rewards = [(f_s - mu_s) / (sigma_s + 1e-10) + lambda_value * (f_u - mu_u) / (sigma_u + 1e-10)
               for f_s, f_u in zip(f_s_list, f_u_list)]
    
    return rewards







# 定義Dataset
class MatchingDataset(Dataset):
    def __init__(self, data, class_to_hypothesis, label_map):
        self.data = []
        self.class_to_hypothesis = class_to_hypothesis
        self.label_map = label_map
        
        for label, text in data:
            for hyp_label, hypothesis in class_to_hypothesis.items():
                self.data.append((text, hypothesis, 1 if label == hyp_label else 0))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)






# 定義BERT-based matching model
class BertMatchingModel(nn.Module):
    def __init__(self, max_length=256):
        super(BertMatchingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 1) # 線性分類器: 輸入維度為768, 輸出維度為1
        self.max_length = max_length 
    # 定義模型的前向傳播方法, 接受input_ids和attention_mask作為輸入
    def forward(self, input_ids, attention_mask):
        # 輸入傳遞給BERT模型, 獲取輸出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]  # 取出[CLS]對應的hidden stae
        # [CLS] token表示傳入線性分類器, 得到logits
        logits = self.classifier(cls_output)
        # 對logits應用sigmoid函數, 輸出壓縮至0-1之間, 返回最終的預測機率(二元分類任務)
        return torch.sigmoid(logits)

# 準備訓練數據
def prepare_data(batch, tokenizer, max_length=256):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text, hypothesis, label in batch:
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
            input_id = torch.cat([encoded['input_ids'].squeeze(), torch.zeros(padding_length, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)], dim=0)
        else:
            input_id = encoded['input_ids'].squeeze()[:max_length]
            attention_mask = torch.ones(max_length, dtype=torch.long)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return input_ids, attention_masks, labels

# 預測函式
def predict(model, dataloader, tokenizer, max_length, device):
    model.eval()
    confidences = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Predicting", unit="batch")
        for batch in progress_bar:
            input_ids, attention_masks, _ = prepare_data(batch, tokenizer, max_length)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            outputs = model(input_ids, attention_masks)
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            for (_, text, _), probs in zip(batch, probabilities):
                confidences.append((text, probs))
            
            # 更新進度描述
            progress_bar.set_postfix({'Processed': f'{len(confidences)}'})
    
    return confidences






def create_label_map(train_dataset, class_to_hypothesis):
    all_labels = set([label for label, _ in train_dataset] + list(class_to_hypothesis.keys()))
    return {label: idx for idx, label in enumerate(sorted(all_labels))}











def main():
    
    # 訓練資料路徑
    train_file = './datasets/emotion/train_pu_half_v0.txt'
    unseen_data_file = './datasets/emotion/train_pu_half_v1.txt'
    dev_file = './datasets/emotion/dev.txt'

    # 建立Dataset
    train_dataset = EmotionDataset(train_file)
    unseen_dataset = EmotionDataset(unseen_data_file)
    D_s_dev_dataset = EmotionDataset(dev_file)

    # 初始化policy network
    policy_network = PolicyNetwork(768 + 1, 128, 2)
    optimizer_policy = torch.optim.Adam(policy_network.parameters(), lr=1e-3)
    return

    # 假設有以下seen classes和對應的hypothesis(train_pu_half_v0.txt訓練集的標籤描述)
    class_to_hypothesis = {
        'shame': 'This text expresses shame.',
        'sadness': 'This text expresses sadness.',
        'love': 'This text expresses love.',
        'fear': 'This text expresses fear.',
        'anger': 'This text expresses anger.',
    }

    label_map = create_label_map(train_dataset, class_to_hypothesis)
    print(f'label_map: {label_map}')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 找訓練集文本最大長度的那個文本, 當作最大文本長度
    max_text_len = max(len(tokenizer.encode(text)) for _, text in train_dataset)
    max_hyp_len = max(len(tokenizer.encode(hyp)) for hyp in class_to_hypothesis.values())
    print(f"Max text length: {max_text_len}, Max hypothesis length: {max_hyp_len}")
    max_length = min(512, max_text_len + max_hyp_len + 3)  # +3 for [CLS] and [SEP] tokens
    print(f"Using max_length: {max_length}")

    # 初始化model和tokenizer
    model = BertMatchingModel(max_length=max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    

    # 初始化優化器和損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()

    # 設定超參數
    num_iterations = 5
    num_episodes = 20
    lambda_value = 1.0
    delta = 0.2  # delta控制子集的大小
    early_stop_patience = 3  



    # Initialize pseudo-labeled data D^p
    D_p = []
    best_f1 = 0
    no_improvement_count = 0

    for i in range(num_iterations):
        # 訓練匹配模型 f
        train_data = [(label, text) for label, text in train_dataset] + [(label, text) for label, text in D_p]
        unseen_data = [(label, text) for label, text in unseen_dataset]
        train_dataloader = DataLoader(MatchingDataset(train_data, class_to_hypothesis, label_map), 
                          batch_size=16, shuffle=True, collate_fn=lambda x: x, num_workers=12)
        unseen_dataloader = DataLoader(MatchingDataset(unseen_data, class_to_hypothesis, label_map), 
                               batch_size=16, shuffle=False, collate_fn=lambda x: x)

        

        for batch in train_dataloader:
            print(f"Batch type: {type(batch)}, length: {len(batch)}")
            print(f"First item in batch: {batch[0]}")
            input_ids, attention_masks, labels = prepare_data(batch, tokenizer, max_length)
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention Masks shape: {attention_masks.shape}")
            print(f"Labels shape: {labels.shape}")

            for idx, seq in enumerate(input_ids):
                print(f"Sequence {idx} length: {(seq != 0).sum()}")
                break

        print(f"\nIteration {i+1}/{num_iterations}")
        train(model, optimizer, criterion, train_dataloader, tokenizer, max_length, device)


        model.eval()

        # 在 D_u 數據集上進行預測
        confidences = predict(model, unseen_dataloader, tokenizer, max_length, device)
        print(f'First few confidences: {confidences[:5]}')
       
        
        """
        # 選擇子集 Ω
        confidences.sort(key=lambda x: max(x[2]), reverse=True)
        subset_size = int(len(confidences) * delta)
        Omega = confidences[:subset_size]

        D_p_i = []
        f_s_list, f_u_list = [], []

        for j in range(num_episodes):
            random.shuffle(Omega)
            batches = [Omega[k:k+10] for k in range(0, len(Omega), 10)]  # 動態批次大小

            episode_states, episode_actions, episode_rewards = [], [], []

            for B_k in batches:
                B_p_k = []
                batch_states, batch_actions = [], []

                for true_label, text, probabilities in B_k:
                    cls_embeddings, _ = predict(text, class_to_hypothesis.keys(), model, tokenizer, class_to_hypothesis)
                    state = torch.cat((torch.tensor(cls_embeddings[0]), torch.tensor([max(probabilities)])))
                    action_probs = policy_network(state)
                    action = torch.multinomial(action_probs, num_samples=1).item()
                    
                    batch_states.append(state)
                    batch_actions.append(action)

                    if action == 1:
                        B_p_k.append((true_label, text))

                # 訓練模型f' 使用B_p_k訓練
                if B_p_k:
                    train_data = [(text, label) for label, text in B_p_k]
                    train_dataloader = DataLoader(MatchingDataset(train_data, class_to_hypothesis, label_map), batch_size=16, shuffle=True)
                    train(model, None, optimizer, None, criterion, train_dataloader, tokenizer, num_epochs=1)

                # 評估
                D_s_dev = [(true_label, text, predict(text, class_to_hypothesis.keys(), model, tokenizer, class_to_hypothesis)[1]) 
                           for true_label, text in D_s_dev_dataset]
                D_u_dev = [(true_label, text, predict(text, class_to_hypothesis.keys(), model, tokenizer, class_to_hypothesis)[1]) 
                           for true_label, text in D_p]

                f_s_k, f_u_k = evaluate(model, D_s_dev, D_u_dev, class_to_hypothesis)
                f_s_list.append(f_s_k)
                f_u_list.append(f_u_k)

                episode_states.extend(batch_states)
                episode_actions.extend(batch_actions)

                D_p_i.extend(B_p_k)

            # 計算reward並更新policy network
            rewards = compute_rewards(f_s_list, f_u_list, lambda_value)
            update_policy_network(policy_network, optimizer_policy, episode_states, episode_actions, rewards)

        # 更新D_p和D_u
        D_p.extend(D_p_i)
        unseen_dataset = [item for item in unseen_dataset if item not in D_p_i]

        # 更新D_u_dev
        D_u_dev = D_p

        # early stop檢查
        current_f1 = (f_s_list[-1] + f_u_list[-1]) / 2  # 使用最後一個batch的F1分數
        if current_f1 > best_f1:
            best_f1 = current_f1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop_patience:
            print(f"Early stopping at iteration {i+1}")
            break

    # 儲存訓練好的模型
    torch.save(model.state_dict(), "model.pt"
    """

if __name__ == "__main__":
    main()