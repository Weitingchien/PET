import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader



# 定義Dataset
class MatchingDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]





# 定義BERT-based matching model
class BertMatchingModel(nn.Module):
    def __init__(self):
        super(BertMatchingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]  # 取出[CLS]對應的hidden state
        score = self.linear(cls_output)
        return torch.sigmoid(score)

# 準備訓練數據
def prepare_data(batch, tokenizer):
    texts, labels = batch
    
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    
    return input_ids, attention_masks, labels



# 預測函式
def predict(text, candidate_labels, model, tokenizer, class_to_hypothesis):
    print(f'candidate_labels: {candidate_labels}')
    inputs = []
    for label in candidate_labels:
        hypothesis = class_to_hypothesis[label]
        input_text = f"[CLS] {text} [SEP] {hypothesis} [SEP]"
        inputs.append(input_text)

    labels = [1] * len(inputs)
    input_ids, attention_masks, _ = prepare_data((inputs, labels), tokenizer)
    
    with torch.no_grad():
        scores = model(input_ids, attention_masks)

    
    probabilities = scores.squeeze().tolist()
    return {label: prob for label, prob in zip(candidate_labels, probabilities)}




def main():


    # 假設有以下seen classes和對應的hypothesis
    class_to_hypothesis = {
        'joy': 'This text expresses joy.',
        'sadness': 'This text expresses sadness.',
        'fear': 'An emotion experienced in anticipation of some specific pain or danger.',
        'anger': 'This text expresses anger.',
        'scary': 'This text expresses scary.',
    }
    # 準備一些訓練樣本
    train_samples = [
        # (text, true_label)
        ("I'm so happy today!", 'joy'),
        ("I feel really sad after watching that movie.", 'sadness'),
        ("The sudden noise scared me a lot.", 'fear'),
        ("I'm so mad at you for lying to me!", 'anger'),
        ("It's such a delightful surprise!", 'joy'),
        ("The gloomy weather makes me feel depressed.", 'sadness'),
        ("I'm trembling with fear.", 'fear'),
        ("I'm furious about the decision made by the committee.", 'anger'),
        ("I had a car accident.", 'fear')
    ]

    # 準備一些測試樣本
    test_samples = [
        "I feel so happy today!",
        "I'm really disappointed with the results.",
        "The movie was so scary!"
    ]



    # 依照Figure 3的格式生成訓練數據
    train_data = []
    for text, true_label in train_samples:
        for label, hypothesis in class_to_hypothesis.items():
            input_text = f"[CLS] {text} [SEP] {hypothesis} [SEP]"

            label = 1 if label == true_label else 0
            train_data.append((input_text, label))
    print(f'train_data: {train_data}')
    """
    # 印出生成的訓練數據
    for input_text, label in train_data:
        print(f"Input: {input_text}")
        print(f"Label: {label}")
        print("----------")
    """


    # 將train_data轉換為Dataset和DataLoader
    train_dataset = MatchingDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    """
    for batch in train_dataloader:
        texts, labels = batch
        print(f"Batch texts: {texts}")
        print(f"Batch labels: {labels}")
        print("----------")
    """


    # 初始化model和tokenizer
    model = BertMatchingModel()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    # 訓練loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()
    num_epochs = 50
    with open("training_results.txt", "w") as f:
        for _ in range(num_epochs):
            for batch in train_dataloader:
                input_ids, attention_masks, labels = prepare_data(batch, tokenizer)

                scores = model(input_ids, attention_masks)
                print(f'scores.shape: {scores.shape} scores: {scores} labels: {labels.float()}')
                # print(f'labels: {labels}')
                loss = criterion(scores.squeeze(), labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 將每個樣本的文本、預測機率和loss寫入.txt檔案

                for text, score, label in zip(batch[0], scores.squeeze().tolist(), labels.tolist()):
                    f.write(f"Text: {text}\n")
                    f.write(f"Predicted probability: {score:.4f}\n")
                    f.write(f"True label: {label}\n")
                    f.write(f"Loss: {loss.item():.4f}\n")
                    f.write("----------\n")

    # 儲存訓練好的模型
    torch.save(model.state_dict(), "model.pt")


    # 使用訓練好的模型做預測
    model.eval()
    candidate_labels = list(class_to_hypothesis.keys())

    for text in test_samples:
        label_probabilities = predict(text, candidate_labels, model, tokenizer, class_to_hypothesis)
        print(f"Text: {text}")
        for label, prob in label_probabilities.items():
            print(f"  {label}: {prob:.4f}")
        print("----------")




if __name__ == "__main__":
    main()