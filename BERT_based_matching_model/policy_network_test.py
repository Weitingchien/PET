import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, confidence, bert_representation):
        x = torch.cat([confidence.unsqueeze(0), bert_representation], dim=1)
        x = self.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

def main():
    # 初始化BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # 初始化策略网络
    input_dim = 769  # 768 (BERT representation) + 1 (confidence)
    hidden_dim = 100
    output_dim = 2  # 选择或不选择
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)

    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    def get_bert_prediction(text):
        # 使用旧版本的tokenizer方法
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([tokenizer.build_inputs_with_special_tokens(token_ids)])
        
        with torch.no_grad():
            outputs = bert_model(input_ids)
            logits = outputs[0]
        
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        
        # 获取BERT表示
        with torch.no_grad():
            bert_outputs = bert_model.bert(input_ids)
            bert_representation = bert_outputs[0][:, 0, :]  # [CLS] token
        
        return confidence.item(), bert_representation

    def select_instance(text):
        confidence, bert_representation = get_bert_prediction(text)
        action_probs = policy_net(torch.tensor([confidence]), bert_representation)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs, confidence

    # 模拟获取奖励的过程
    def get_reward(selected_texts):
        # 这里应该是一个更复杂的过程，包括训练分类器和评估
        # 为了简化，我们只返回一个随机奖励
        return torch.rand(1).item()

    # 训练过程
    for episode in range(100):
        selected_texts = []
        log_probs = []
        total_confidence = 0
    
        for text in ["演员的表演令人印象深刻", "剧情太无聊了", "我会向所有人推荐这部电影"]:
            action, action_probs, confidence = select_instance(text)
        
            if action == 1:  # 选择该实例
                selected_texts.append(text)
                total_confidence += confidence
        
            log_probs.append(torch.log(action_probs[0, action]))
    
        reward = get_reward(selected_texts)
    
        # 更新策略网络
        loss = -torch.stack(log_probs).sum() * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if episode % 10 == 0:
            avg_confidence = total_confidence / len(selected_texts) if selected_texts else 0
            print(f"Episode {episode}, Selected {len(selected_texts)} instances, Avg Confidence: {avg_confidence:.4f}, Reward: {reward:.4f}")

if __name__ == "__main__":
    main()