"""
import os
import log
import torch
import argparse
import numpy as np
from typing import List
from utils import InputFeatures
from torch.utils.data import RandomSampler, DataLoader, TensorDataset
from tasks import PROCESSORS, load_examples
from utils import set_seed, eq_div, save_logits, LogitsList, InputExample


import jsonpickle

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

from preprocessor import SequenceClassifierPreprocessor, MLMPreprocessor

from wrapper import TransformerModelWrapper, WRAPPER_TYPES


CONFIG_NAME = 'wrapper_config.json'
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"

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

    def train(self, task_train_data: List[InputExample], device, iteration: int, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_helper_batch_size: int = 8, helper_train_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, **_):

        data_dict = {}

        print(f'len(task_train_data): {len(task_train_data)}')
        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        # _generate_dataset: 將這些InputExample轉換為PyTorch的Dataset,這樣才能被PyTorch的DataLoader使用
        train_dataset = self._generate_dataset(task_train_data)
        print(f'train_dataset: {train_dataset}')
        # RandomSampler的作用是在每個epoch開始時,隨機打亂dataset中樣本的順序。這樣可以確保模型在每個epoch看到的樣本順序都不一樣,有助於減少過擬合
        train_sampler = RandomSampler(train_dataset)
        print(f'train_sampler: {train_sampler}')
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)







def print_train_data_and_all_train_data(train_data, all_train_data):
        # train_data
        for i, val in enumerate(train_data):
            if i > 4:
                break
            print(f"(train_data)Example {i+1}:")
            print(f"  Guid: {train_data[i].guid}")
            print(f"  Text A: {train_data[i].text_a}")
            print(f"  Text B: {train_data[i].text_b}")
            print(f"  Label: {train_data[i].label}")
            print(f"  Logits: {train_data[i].logits}")
            
        # all_train_data
        for i, val in enumerate(all_train_data):
            if i > 4:
                break
            print(f"(all_train_data)Example {i+1}:")
            print(f"  Text: {all_train_data[i].text_a}")
            print(f"  Label: {all_train_data[i].label}")


def simple_accuracy(preds, labels):
    DEPRECATION_WARNING = "This feature is deprecated and may be removed in future versions."
    # warnings.warn(DEPRECATION_WARNING, FutureWarning)
    print(f'(simple_accuracy)preds: {preds}')
    print(f'(simple_accuracy)labels: {labels}')
    return (preds == labels).mean()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_examples", required=True, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(PROCESSORS.keys()))
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="The model type (currently supported are bert and roberta)")
    parser.add_argument("--wrapper_type", required=True, choices=WRAPPER_TYPES,
                        help="The wrapper type - either sequence_classifier (corresponding to"
                             "regular supervised training) or mlm (corresponding to PET training)")
    parser.add_argument("--lm_train_examples_per_label", default=10000, type=int,
                        help="The total number of training examples for auxiliary language modeling, "
                             "where -1 equals all examples")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET training)")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET training)")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to perform lower casing")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET training)")
    parser.add_argument("--logits_file", type=str,
                        help="The logits file for combining multiple PVPs, which can be created using the"
                             "merge_logits.py script")
    args = parser.parse_args()
    args.pattern_id = 0
    args.use_logits = args.logits_file is not None

    args.output_mode = "classification"
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()
    train_examples_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_data = load_examples(args.task_name, args.data_dir, train_examples_per_label, evaluate=False)
    wrapper = TransformerModelWrapper(args)
    all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
    wrapper.train(train_data, device, helper_train_data=all_train_data if args.lm_training or args.use_logits else None, **vars(args))
"""
"""
def eq_div(N, i):
    # Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`.
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)
"""


    # print_train_data_and_all_train_data(train_data,all_train_data)

# if __name__ == "__main__":
"""
    N = 13
    i = 3
    print([N // i + 1] * (N % i))
    print([N // i] * (i - N % i))
    print(13 // 3) # 4
    print(13 %  3) # 1
    sampler = RandomSampler(range(10)) # 打亂原始數據的順序
    print([i for i in sampler]) # [7, 8, 0, 6, 3, 4, 5, 2, 9, 1]
    
    # 3維Tensor
    
    logits = torch.randn(4, 5, 10)
    print(f'logits: {logits}')
    print(f'logits.shape: {logits.shape}')
    
    # list轉換成 tensor      
    
    mlm_labels = torch.full((4, 5), -1)# 建立一個全為-1的張量
    mlm_labels[:, 3] = 1 # 把第4個位置的值設為1,代表被遮蔽位置
    print(f'mlm_labels: {mlm_labels}') # [tensor([False, False, False,  True, False, False, False])]
    masked_logits = logits[mlm_labels >= 0]
    print(f'c: {masked_logits}')

    m2c = torch.tensor([[8], [10], [5], [4], [3]])
    print(f'm2c: {m2c}')

    b = torch.zeros_like(m2c)
    print(f'b: {b}')

    c = torch.max(torch.zeros_like(m2c), m2c)
    print(f'c: {c}')

    d =  (m2c > 0).float()
    print(f'd: {d}')


    # cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
    # print(f'cls_logits: {cls_logits}')
    # main()
    
    merge_logits.py: np.mean說明
    
    test_logits = np.array([
        [5.9, -2.2, -4.3, 2.3, 19.8],
        [4.2, -4.7, -4.2, -4.1, 16.2],
        [5.4, -2.1, -4.7, 3.1, 20.8],
        [2, -5, -3.5, -1.1, 18]])

    logits = np.array([
        # p0-i0
        [5.9478846, -2.246691, -4.30881, 2.3847644, 19.828459],
        # p0-i1
        [5.4069896, -3.756437, -3.2553425, 3.62263, 20.259268], 
        # p0-i2
        [4.4773693, -3.9174242, -3.3116395, 3.76114, 20.80512],
        # p1-i0
        [4.2726197, -4.791045, -4.2288995, -4.120239, 16.28123],
        # p1-i1
        [4.611694, -3.070371, -5.9071527, -1.001478, 18.717123],
        # p1-i2
        [5.837471, -3.7982347, -2.9907975, 3.5469222, 20.82018],
        # p2-i0
        [5.4967856, -2.1803536, -4.707951, 3.189568, 20.856955],
        # p2-i1
        [4.488108, -3.852061, -5.194917, 4.3645945, 19.91591],
        # p2-i3
        [5.260121, -2.7068188, -6.505483, 1.3350872, 19.978271],
        # p3-i0
        [2.060473, -5.0971932, -3.5274274, -1.1608342, 18.032104],
        # p3-i1
        [4.350354, -1.9898229, -3.1502078, 1.4578516, 18.866089],
        # p3-i2
        [2.7231414, -3.5293348, -4.3000526, 0.75283766, 19.091772]
    ])

    e = np.mean(test_logits, axis=0).tolist()
    print(f'e: {e}') # e: [5.0, 6.0, 7.0, 8.0]
    # print(f'eq_div: {eq_div(1, 5)}')
    preds = np.array([3, 2, 3, 3, 3, 3])
    labels = np.array([4, 1, 3, 3, 0, 4])
    print(f'simple_accuracy: {simple_accuracy(preds, labels)}')



    CLASStorch.utils.data.RandomSampler(data_source,
    replacement=False, num_samples=None, generator=None)

    data_source: 被採樣的數據集
    replacement: 採樣策略, 預設為False, 如果為True 表示可重複對一個樣本採樣, False為一個樣本最多只能被採樣一次
    num_samples: 默認取全部樣本,如果replacement為True時, 可指定採樣數量, 透過修改num_samples值
    generator: 採樣過程中的生成器
"""


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
This script can be used to generate a training set for the next generation in iPET
using the previous generation of models.
"""
import argparse
import ast
import os
import random
from copy import deepcopy
from typing import List
import numpy as np

import log
import utils
from utils import InputExample, LogitsList
from run_training import load_examples, eq_div
from tasks import PROCESSORS

logger = log.get_logger('root')


def generate_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                       num_examples: int, logits_percentage: float, reduction: str = 'mean',
                       n_most_likely: int = -1) -> List[InputExample]:
    """
    Generate a training set for the next generation of iPET models
    :param logits_lists: predictions from the previous generation of models
    :param labels: all task labels
    :param original_data: the original training data corresponding to the logits_lists
    :param num_examples: the number of examples to create
    :param logits_percentage: the percentage of models/logits to choose
    :param reduction: the reduction strategy ('wmean' or 'mean')
    :param n_most_likely: if >0, for each label the n_most_likely examples with the highest logits are chosen
    :return: a list of input examples that serves as training set for the next generation
    """
    # Ensures that all LogitsList objects have the same number of logits.
    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    num_logits_lists = round(len(logits_lists) * logits_percentage) # 根據logits_percentage計算要選擇的logits_lists的數量,然後從logits_lists中隨機選擇該數量的元素
    """
        Suppose we have 4 LogitsList objects, and logits_percentage is 0.5.
        This results in selecting 2 random LogitsList objects.
    """
    logits_lists = random.sample(logits_lists, k=num_logits_lists)
    # Converts logits and scores into numpy arrays for processing.
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0)
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)
    logits = utils.softmax(logits, axis=1).tolist()

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]
    examples_per_label = eq_div(num_examples, len(labels))

    test_set = []

    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info("There are {} examples for label {}".format(len(examples), label))
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples,
            num_examples=examples_per_label[idx])
        test_set.extend(label_examples)
    print(f'test_set: {test_set}')
    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples]
    sum_label_probabilities = sum(label_probabilities)
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    return np.random.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logits_dir", type=str, required=True,
                        help="The dir in which the results of all PVPs are stored in separate subdirs. "
                             "Each subdir is expected to have a file 'results.txt' and a file 'logits.txt' in "
                             "it as created by 'run.py'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The dir where the generate training sets are to be saved.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Whether to overwrite the output dir's content if it already exists")
    parser.add_argument("--reduction", required=True, choices=['mean', 'wmean'],
                        help="The reduction strategy for merging logits. Must be one of 'mean' or 'wmean', "
                             "where the latter is short for 'weighted mean' and the weights for each PVP are "
                             "proportional to its score on the training set before fine-tuning.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")
    parser.add_argument("--lm_train_examples_per_label", required=True, type=int,
                        help="The number of unlabeled examples per label that were annotated using the previous "
                             "generation of models")
    parser.add_argument("--num_examples", required=True, type=int,
                        help="The total number of examples to create")
    parser.add_argument("--logits_percentage", required=True, type=float,
                        help="The percentage of logits (i.e., models) to choose for generating the next training sets")
    parser.add_argument("--seed", default=42, type=int, help="RNG seed")
    parser.add_argument("--n_most_likely", type=int, default=-1)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    processor = PROCESSORS[args.task_name]()
    labels = processor.get_labels()

    subdirs = next(os.walk(args.logits_dir))[1] # 獲取logits_dir目錄下的所有子目錄

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
    # 加載所有的訓練數據,並將每個樣本的label和logits屬性設置為None。
    for example in all_train_data:
        example.label = None
        example.logits = None

    logits_lists = {}

    for subdir in subdirs:
        results_file = os.path.join(args.logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(args.logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        # 從results.txt中讀取訓練前的accuracy
        with open(results_file, 'r') as fh:
            results = ast.literal_eval(fh.read())
            result_train = results['train_set_before_training']

        # 從logits.txt中讀取logits,每列(row)表示一個樣本的logits,將其轉換為浮點數
        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info(
            f"File {results_file}: Score = {result_train}, #Logits = {len(logits)}, #Labels = {len(logits[0])}"
        )

        loglist = LogitsList(score=result_train, logits=logits)
        logits_lists[subdir] = loglist

    for subdir in subdirs:
        other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        subdir_train_set = generate_train_set(other_logits_lists,
                                              labels=labels,
                                              original_data=all_train_data,
                                              num_examples=args.num_examples,
                                              logits_percentage=args.logits_percentage,
                                              reduction=args.reduction,
                                              n_most_likely=args.n_most_likely)

        InputExample.save_examples(
            subdir_train_set,
            os.path.join(args.output_dir, f'{subdir}-train.txt'),
        )


if __name__ == "__main__":
    main()















