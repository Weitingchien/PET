# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import warnings
import jsonpickle
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, BertForMaskedLM, \
    RobertaForMaskedLM, XLMRobertaForMaskedLM
from transformers import (BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMRobertaConfig,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer,
                          )
from transformers import __version__ as transformers_version
# from transformers.data.metrics import simple_accuracy

import log
from preprocessor import SequenceClassifierPreprocessor, MLMPreprocessor
from utils import InputFeatures

logger = log.get_logger('root')

CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: SequenceClassifierPreprocessor,
    MLM_WRAPPER: MLMPreprocessor
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification,
        MLM_WRAPPER: RobertaForMaskedLM
    },
    'xlm-roberta': {
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLMRobertaForSequenceClassification,
        MLM_WRAPPER: XLMRobertaForMaskedLM
    }
}



"""
    假設preds == labels的結果是[True, False, True, True, False]
    將True轉換為1,False轉換為0,得到[1, 0, 1, 1, 0]
    計算平均值,得到(1 + 0 + 1 + 1 + 0) / 5 = 0.6
"""
def simple_accuracy(preds, labels):
    DEPRECATION_WARNING = "This feature is deprecated and may be removed in future versions."
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    print(f'(simple_accuracy)preds: {preds}')
    print(f'(simple_accuracy)labels: {labels}')
    return (preds == labels).mean()




def distillation_loss(predictions, targets, temperature):
    """
        predictions: 學生模型的預測結果
        targets: 教師模型的預測結果
        temperature: ,控制softmax函數的平滑度
        第一步: 將predictions(學生模型的預測結果)和targets(教師模型的預測結果)通過softmax函數轉換為機率分佈
    """
    print(f'predictions: {predictions} targets: {targets}')
    p = F.log_softmax(predictions / temperature, dim=1)
    print(f'p: {p}')
    q = F.softmax(targets / temperature, dim=1)
    print(f'q: {q}')
    print(f'predictions.shape[0]: {predictions.shape[0]}')
    return F.kl_div(p, q, reduction='sum') * (temperature ** 2) / predictions.shape[0]


class WrapperConfig(object):
    def __init__(self, model_type, wrapper_type, task_name, max_seq_length: int, label_list: List[str],
                 pattern_id: int, verbalizer_file: str):
        self.model_type = model_type
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.verbalizer_file = verbalizer_file


class TransformerModelWrapper:

    def __init__(self, args):
        self.config = WrapperConfig(
            model_type=args.model_type, wrapper_type=args.wrapper_type, task_name=args.task_name,
            max_seq_length=args.max_seq_length, label_list=args.label_list, pattern_id=args.pattern_id,
            verbalizer_file=args.verbalizer_file
        )

        config_class = MODEL_CLASSES[self.config.model_type]['config']
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(
            args.model_name_or_path, num_labels=len(args.label_list), finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None)  # type: PreTrainedTokenizer

        self.model = model_class.from_pretrained(args.model_name_or_path, config=model_config,
                                                 cache_dir=args.cache_dir if args.cache_dir else None)

        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](self, self.config.task_name, self.config.pattern_id,
                                                                    self.config.verbalizer_file)

    @classmethod
    def from_pretrained(cls, path: str) -> 'TransformerModelWrapper':
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](
            wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)
        return wrapper

    def save(self, path: str) -> None:
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    def _load_config(self, path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    def print_unlabeled_examples(self, unlabeled_data: List[InputExample], num_examples: int = 10):
        print(f"Printing {num_examples} unlabeled examples, Total: {len(num_examples)}")
        for i, example in enumerate(unlabeled_data[:num_examples], start=1):
            print(f"Example {i}:")
            print(f"  Text: {example.text_a}")
            if example.text_b is not None:
                print(f"  Text B: {example.text_b}")
            print(f"  Label: {example.label}")
            print()

    def train(self, task_train_data: List[InputExample], device, iteration: int, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_helper_batch_size: int = 8, helper_train_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, **_):

        print(f'(train) lm_training: {lm_training}')
        print(f'(train) use_logits: {use_logits}') # (train) use_logits: False


        data_dict = {}

        print(f'len(task_train_data): {len(task_train_data)}')
        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        print(f'train_batch_size: {train_batch_size}') # 4
        # _generate_dataset: 將這些InputExample轉換為PyTorch的Dataset,這樣才能被PyTorch的DataLoader使用
        train_dataset = self._generate_dataset(task_train_data)
        # RandomSampler的作用是在每個epoch開始時,隨機打亂dataset中樣本的順序。這樣可以確保模型在每個epoch看到的樣本順序都不一樣,有助於減少過擬合
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)


        helper_dataloader, helper_iter = None, None

        if lm_training or use_logits:
            for i in range(5):
                print(f'開始執行helper_dataloader')
            assert helper_train_data is not None
            helper_batch_size = per_gpu_helper_batch_size
            # labelled=False => mlm_labels = [-1] * self.wrapper.config.max_seq_length
            helper_dataset = self._generate_dataset(helper_train_data, labelled=False)
            helper_sampler = RandomSampler(helper_dataset)
            helper_dataloader = DataLoader(helper_dataset, sampler=helper_sampler, batch_size=helper_batch_size)
            """
            for i in range(3):
                batch = next(iter(helper_dataloader))
                print(f"(helper_dataloader) Batch {i+1}:")
                for j in range(len(batch)):
                    print(f"  {j}: {batch[j]}")
            """

            helper_iter = helper_dataloader.__iter__()

        if use_logits:
            train_dataloader = helper_dataloader

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        t_epoch = len(train_dataloader) // gradient_accumulation_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad() # 將模型梯度歸零
        """
            trange: 用於為一個範圍內的迭代顯示進度條
            int(num_train_epochs): 迭代的總次數,即訓練的總週期數
            desc="Epoch": 在進度條前方顯示的字串為"Epoch"
            時間格式: [已運行時間<預估剩餘時間, 每次迭代平均耗時]
        """
        train_iterator = trange(int(num_train_epochs), desc="Epoch")



        # 外層的train_iterator: 整個訓練的epoch進度
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            # 每個epoch內數據載入的進度
            for step, batch in enumerate(epoch_iterator):
                self.model.train()

                if lm_training:
                    helper_batch = None
                    while helper_batch is None:
                        try:
                            helper_batch = helper_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting helper batch")
                            helper_iter = helper_dataloader.__iter__()
                print(f'batch[0]: {batch[0]}')
                decoded_inputs = [self.tokenizer.decode(seq_ids) for seq_ids in batch[0]]
                print(f'decoded_inputs: {decoded_inputs}')
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.config.model_type in ['bert', 'xlnet'] else None}

                if self.config.wrapper_type == SEQUENCE_CLASSIFIER_WRAPPER and not use_logits:
                    inputs['labels'] = batch[3]

                inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
                print(f'inputs: {inputs}')
                # **inputs: 參數解包
                outputs = self.model(**inputs) # outputs[0].shape: torch.Size([4, 256, 50265])
                # print(f'outputs[1]: {outputs[1]}')
                # print(f'outputs[0]: {outputs[0]}')
                
                for i, output in enumerate(outputs):
                    print(f'outputs[{i}].shape: {output.shape}')
                



                if use_logits:
                    print("Using distillation training with logits")
                    logits_predicted = outputs[0]
                    print(f'logits_predicted: {logits_predicted}')
                    # 這個logits是每個模型預測後,算術平均的值,利用這個值來去計算loss 
                    logits_target = batch[5].to(device)
                    print(f'logits_target: {logits_target}')
                    loss = distillation_loss(logits_predicted, logits_target, temperature)
                    print(f'distillation_loss: {loss}')
                    """
                    print(f"Predicted logits shape: {logits_predicted.shape}")
                    print(f"Target logits shape: {logits_target.shape}")
                    """
                elif self.config.wrapper_type == SEQUENCE_CLASSIFIER_WRAPPER:
                    for _ in range(4):
                        print("Training for sequence classification task")
                    loss = outputs[0]
                    for i in range(5):
                    
                        print(f"Loss value: {loss}")
                    
                else:
                    for _ in range(4):
                        print("Training for MLM task")
                    #print()
                    

                    mlm_labels = batch[4].to(device)
                    # print(f'(train)mlm_labels: {mlm_labels}, shape: {mlm_labels.shape}')
                    # 用於將 MLM 的預測分數轉換為分類任務的預測分數(利用 MLM 任務學習到的知識來執行分類任務)
                    prediction_scores, predicted_words = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels,
                                                                                               outputs[0], data_dict, iteration)
                    # print(f'prediction_scores.shape:  {prediction_scores.shape}')  # torch.Size([4, 5])
                    input_ids = batch[0] 
                    decoded_inputs = [self.tokenizer.decode(ids) for ids in input_ids]
                    # print(f'(train) decoded_inputs: {decoded_inputs} ')
                    labels = batch[3].to(device) #真實標籤
                    print(f'(train)prediction_scores: {prediction_scores} predicted_words: {predicted_words} decoded_inputs: {decoded_inputs} labels: {labels}')
                    loss_fct = nn.CrossEntropyLoss()
                    for i in range(0, 10):
                        print(f'prediction_scores: {prediction_scores}')
                        print(f'prediction_scores.view(): {prediction_scores.view(-1, len(self.config.label_list))} labels.view(-1):{labels.view(-1)}')
                    loss = loss_fct(prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1))
                    for i in range(0, 10):
                         print(f'loss: {loss}')
                    """
                    print(f"MLM logits shape: {outputs[0].shape}")
                    print(f"Prediction scores shape: {prediction_scores.shape}")
                    print(f"Labels shape: {labels.shape}")
                    """

                if lm_training:
                    for i in range(5):
                        print(f'lm_training')
                    lm_inputs = {
                        'input_ids': helper_batch[0], 'attention_mask': helper_batch[1],
                        'masked_lm_labels': helper_batch[4],
                        'token_type_ids': helper_batch[2] if self.config.model_type in ['bert', 'xlnet'] else None}

                    lm_inputs['input_ids'], lm_inputs['masked_lm_labels'] = self._mask_tokens(lm_inputs['input_ids'])
                    lm_inputs = {k: v.to(device) if v is not None else None for k, v in lm_inputs.items()}
                    lm_outputs = self.model(**lm_inputs)
                    lm_loss = lm_outputs[0]
                    loss = alpha * loss + (1 - alpha) * lm_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # gradient_accumulation_steps: 用於控制每隔多少步更新一次模型參數
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                # 對損失函數進行反向傳播,計算梯度
                loss.backward()
                # 將當前batch的損失值累加到tr_loss
                tr_loss += loss.item()
                """
                    檢查是否到了需要更新模型參數的步驟
                    gradient_accumlation_steps: 控制了梯度累積的步數(即每隔多少步更新一次模型參數)
                """
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 對模型的梯度進行裁剪(可以防止梯度爆炸)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    # 優化器根據計算出的梯度更新模型參數
                    optimizer.step()
                    # 學習率調度器根據設定的策略調整學習率
                    scheduler.step()
                    # 模型的梯度清零,為下一次梯度計算做準備
                    self.model.zero_grad()
                    # 用於記錄訓練的總步數
                    global_step += 1
                    # 這行程式碼檢查是否到了需要記錄日誌的步驟
                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        # 計算過去logging_steps步的平均損失
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        # 獲取當前的學習率
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        # 轉換為JSON格式並列印出來,用於記錄訓練過程
                        print(json.dumps({**logs, **{'step': global_step}}))
                # 檢查當前的global_step是否超過了設定的最大訓練步數max_steps(如果超過了,就提前結束當前的epoch或整個訓練過程)
                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return data_dict, global_step, (tr_loss / global_step if global_step > 0 else -1)

    def eval(self, eval_data: List[InputExample], device, per_gpu_eval_batch_size: int = 8, n_gpu: int = 1,
             output_logits: bool = False, **_):

        print(f'len(eval_data): {len(eval_data)}')
        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        print(f'len(eval_dataloader): {len(eval_dataloader)}')

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # print(f'batch: {batch}')
            self.model.eval()
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            labels = batch[3]
            mlm_labels = batch[4]


        
            # 解碼輸入序列
            decoded_inputs = [self.tokenizer.decode(seq_ids) for seq_ids in input_ids]

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.config.model_type in ['bert', 'xlnet'] else None}
                outputs = self.model(**inputs)
                logits = outputs[0] # (batch_size, sequence_length, vocab_size)=> ([8, 256, 30522])



                if self.config.wrapper_type == MLM_WRAPPER:
                    # 獲取被遮蔽位置的索引
                    masked_indices = (mlm_labels == 1).nonzero(as_tuple=True)

                    for seq_idx in range(input_ids.size(0)):
                        seq_mlm_labels = mlm_labels[seq_idx]
                        seq_masked_indices = (seq_mlm_labels == 1).nonzero(as_tuple=True)[0] #nonzero:回傳非0元素的位置
                        seq_logits = logits[seq_idx]  # (sequence_length, vocab_size)
                        seq_input_ids = input_ids[seq_idx]  # (sequence_length)
                        
                        for pos in seq_masked_indices:
                            # 獲取該位置的預測分數
                            token_logits = seq_logits[pos]  # (vocab_size)

                            # 找到分數最高的前幾個單字的id
                            # torch.topk()會返回seq_logits在最後一個維度上的前k個最大值及其對應的索引
                            top_k_ids = torch.topk(token_logits, k=5).indices.tolist() 

                    
                            # 將id轉成單字
                            top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_ids)
                            top_k_scores = token_logits[top_k_ids].tolist()
                            # print(f"mlm_labels: {np.array2string(mlm_labels.cpu().numpy(), separator=', ', threshold=np.inf)}")
                            # print(f"Encoded input: {seq_input_ids.tolist()}")
                            # print(f"Decoded input: {decoded_inputs[seq_idx]}")
                            # print(f"Masked position: {pos}")

                            # 將MLM logits轉換成分類logits
                            cls_logits, predicted_words = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels[seq_idx].unsqueeze(0), seq_logits.unsqueeze(0))
            
                            # 獲取分類標籤
                            label_list = self.config.label_list
                            predicted_label_idx = torch.argmax(cls_logits, dim=-1).item()
                            predicted_label = label_list[predicted_label_idx]

                            # print(f"Predicted label: {predicted_label}")
                            print(f'predicted_words: {predicted_words}')


                            for token, score in zip(top_k_tokens, top_k_scores):
                                """
                                print(f"Predicted token: {token}, Score: {score:.4f}")
                                """
                                pass
                                
                            
                            # print()

                    logits, predicted_words = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, logits)
            nb_eval_steps += 1
            """
            if nb_eval_steps == 1:
                # 打印第一筆資料的原始輸入文本
                print("First example - Original Text:")
                first_example = eval_data[0]
                print(first_example.text_a)
                if first_example.text_b is not None:
                    print(first_example.text_b)
                print()

                print(f"Example {nb_eval_steps} - Decoded Inputs:")
                decoded_input = self.tokenizer.decode(batch[0][nb_eval_steps-1])
                print(decoded_input)
                print()

                # 打印第一筆資料的encode後的輸入數據
                print("First example - Encoded Inputs:")
                print("input_ids:", batch[0][0])
                print("attention_mask:", batch[1][0])
                print("token_type_ids:", batch[2][0] if self.config.model_type in ['bert', 'xlnet'] else None)
                print()
            """

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()

                if nb_eval_steps == 1:
                    # 打印第一筆資料的預測標籤和真實標籤
                    print(f"First example - Predicted label: {np.argmax(preds[0])}")
                    print(f"First example - True label: {out_label_ids[0]}")
                    print()
                    print(f'preds.shape: {preds.shape}')
                    print(f'out_label_ids: {out_label_ids}')
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0) # 儲存模型對所有評估樣本的預測結果
                # print(f'preds: {preds}')
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0) # 包含了所有評估樣本的真實標籤
                print(f'out_label_ids: {out_label_ids}')

        if output_logits:
            return preds
        
        preds = np.argmax(preds, axis=1) # 找到最大值得索引(沿著row方向)
        print(f'np.argmax(preds, axis=1): {preds}')
        return {"acc": simple_accuracy(preds, out_label_ids)}

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)
        all_logits = torch.tensor([f.logits for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_mlm_labels,
                                all_logits)
        return dataset

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info(f"Writing example {ex_index}")
            input_features = self.preprocessor.get_input_features(example, labelled=labelled)
            features.append(input_features)
        return features

    def _mask_tokens(self, input_ids):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # if a version of transformers < 2.4.0 is used, -1 is the expected value for indices to ignore
        if [int(v) for v in transformers_version.split('.')][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1

        labels[~masked_indices] = ignore_value  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels
