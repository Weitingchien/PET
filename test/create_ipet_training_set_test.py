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

import logging
import utils
from utils import InputExample, LogitsList
from run_training import load_examples, eq_div
from tasks import PROCESSORS

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def generate_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                       num_examples: int, logits_percentage: float, reduction: str = 'mean',
                       n_most_likely: int = -1) -> List[InputExample]:
    print(f'len(logits_lists): {len(logits_lists)}') # 11
    print(f'len(original_data): {len(original_data)}') # len(original_data): 50000
    print(f'num_examples: {num_examples}') # num_examples: 100
    print(f'n_most_likely: {n_most_likely}') # n_most_likely: -1
    print(f'logits_percentage: {logits_percentage}') # logits_percentage: 0.25
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
    # LogitsList objects: 每個pattern都會有對應的LogitsList objects(logits.txt), 
    # Ensures that all LogitsList objects have the same number of logits.
    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    num_logits_lists = round(len(logits_lists) * logits_percentage) # 根據logits_percentage計算要選擇的logits_lists的數量,然後從logits_lists中隨機選擇該數量的元素
    # print(f'num_logits_lists: {num_logits_lists}') # num_logits_lists: 3
    
    """
        Suppose we have 4 LogitsList objects, and logits_percentage is 0.5.
        This results in selecting 2 random LogitsList objects.
    """
    logits_lists = random.sample(logits_lists, k=num_logits_lists) # k=3, random.sample:隨機選取3個LogitsList objects(logits_lists已經去除當前pattern的logits,假設當前subdir是p2-i2,則先去除p2-i2的logits.txt再隨機選取3個pattern的LogitsList objects)
    # print(f'logits_lists: {logits_lists}') # logits_lists: [LogitsList(score=0.38, logits[:2]=[[5.837471, -3.7982347, -2.9907975, 3.5469222, 20.82018], [4.035328, 20.567448, 3.0972004, 3.0757291, -4.032412]]), LogitsList(score=0.31, logits[:2]=[[2.7231414, -3.5293348, -4.3000526, 0.75283766, 19.091772], [2.283721, 22.406498, 5.0643315, 6.0301456, -4.6486263]]), LogitsList(score=0.31, logits[:2]=[[2.060473, -5.0971932, -3.5274274, -1.1608342, 18.032104], [3.3292046, 20.60177, 0.24962683, 4.426136, -3.725746]])]
    
    # Converts logits and scores into numpy arrays for processing.
    logits = np.array([ll.logits for ll in logits_lists])
    print(f'len(logits): {len(logits)}') # len(logits): 3
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0) #3個不同pattern的logits從column方向平均(範例: test_numpy.py)
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)
    
    logits = utils.softmax(logits, axis=1).tolist() #針對row方向作正規化(轉換為對每個類別的概率)

    # original_data的前5萬筆訓練樣本是使用3個pattern的平均機率
    for lgs, example in zip(logits, original_data): # 假設最後一筆是train-55330: lgs: 先前3個pattern作算術平均再經softmax轉成的機率值
        # print(f'lgs: {lgs}') # lgs: [2.7359291778355936e-09, 4.866805810314291e-09, 6.360771218671432e-09, 0.4216283643498055, 0.5783716216866882]
        # print(f'labels: {labels}') # ['1', '2', '3', '4', '5']
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()] #np.argmax(example.logits).item(): 把機率最高的值轉換成對應的標籤(根據logits向量的最大值索引,找到對應的標籤)
    
    examples_per_label = eq_div(num_examples, len(labels)) 
    # print(f'examples_per_label: {examples_per_label}') # examples_per_label: [20, 20, 20, 20, 20]

    test_set = []

    for idx, label in enumerate(labels):
        """
            如果 examples 列表中的樣本數量小於 examples_per_labelidx,則通過上採樣(upsampling)的方式補充樣本,
            直到樣本數量達到要求。上採樣是指重複使用標籤為 label 的樣本,以增加樣本數量
        """

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logger.info("There are {} examples for label {}".format(len(examples), label))
            """
                INFO:root:There are 6823 examples for label 1
                len(label_probabilities): 6823
                INFO:root:There are 11801 examples for label 2
                len(label_probabilities): 11801
                INFO:root:There are 10999 examples for label 3
                len(label_probabilities): 10999
                INFO:root:There are 11704 examples for label 4
                len(label_probabilities): 11704
                INFO:root:There are 8673 examples for label 5
                len(label_probabilities): 8673
            """
            
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True) # 對 examples 列表按照 logit 值進行降序排序
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]] # 選擇 logit 值最高的前 n_most_likely 個樣本,儲存在 examples 列表中
            examples = [deepcopy(ex) for ex in examples] # 使用深拷貝(deepcopy)創建選定樣本的副本,以避免修改原始樣本
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples,
            num_examples=examples_per_label[idx])
        test_set.extend(label_examples)
    # print(f'len(test_set): {len(test_set)}') # len(test_set): 100
    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int) -> List[InputExample]:
    label_probabilities = [max(example.logits) for example in examples] # 假設label_probabilities = [0.9, 0.8, 0.7, 0.6]
    print(f'len(label_probabilities): {len(label_probabilities)}')
    sum_label_probabilities = sum(label_probabilities) # 0.9 + 0.8 + 0.7 + 0.6 = 3.0
    label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
    """
        Normalizes the label probabilities so that they sum up to 1,
        label_probabilities = [0.9 / 3.0, 0.8 / 3.0, 0.7 / 3.0, 0.6 / 3.0]
        Result: [0.3, 0.2667, 0.2333, 0.2]
    """
    return np.random.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()
    """
        Uses np.random.choice to randomly select num_examples from examples based on the normalized label_probabilities.
        examples: 從examples抽取
        size=num_examples: 要抽取的樣本數量
        replace=False: 確保樣本不會被重複選擇
        p=label_probabilities: 參數決定了每個樣本被選中的機率。機率值越高，樣本被選中的可能性就越大
    """ 


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
    # print(f'processor: {processor}') # processor: <tasks.YelpFullProcessor object at 0x70ffc0eb2c10>
    labels = processor.get_labels()
    # print(f'labels: {labels}') # labels: ['1', '2', '3', '4', '5']

    subdirs = next(os.walk(args.logits_dir))[1] # 獲取logits_dir目錄下的所有子目錄
    # print(f'subdirs: {subdirs}') # subdirs: ['p2-i2', 'p3-i0', 'p3-i2', 'p1-i1', 'p1-i0', 'p0-i2', 'p0-i1', 'p2-i1', 'p3-i1', 'p0-i0', 'p2-i0', 'p1-i2']
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
    # print(f'len(all_train_data): {len(all_train_data)}') # 50000
    # 加載所有的訓練數據,並將每個樣本的label和logits屬性設置為None(在論文中表示未標籤數據集D)
    for example in all_train_data:
        example.label = None
        example.logits = None
    

    logits_lists = {}

    for subdir in subdirs:
        """
        if subdir != 'p2-i2':
            break
        """

        print(f'subdir: {subdir}') # p2-i2
        results_file = os.path.join(args.logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(args.logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logger.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        # 從results.txt中讀取訓練前的accuracy
        with open(results_file, 'r') as fh:
            results = ast.literal_eval(fh.read())
            # print(f'results: {results}') # results: {'train_set_before_training': 0.25, 'train_set_after_training': 1.0, 'test_set_after_training': 0.5491}
            result_train = results['train_set_before_training']


        # 從logits.txt中讀取logits,每列(row)表示一個樣本的logits,將其轉換為浮點數
        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                # print(f'example_logits: {example_logits}') # example_logits: [5.260121, -2.7068188, -6.505483, 1.3350872, 19.978271]
                logits.append(example_logits)

        # print(f'logits: {logits[:2]}') # logits: [[5.260121, -2.7068188, -6.505483, 1.3350872, 19.978271], [3.1418624, 18.523603, 1.783086, 2.1958969, -1.2012017]]

        logger.info(
            f"File {results_file}: Score = {result_train}, #Logits = {len(logits)}, #Labels = {len(logits[0])}"
        ) # INFO:root:File ./output_roberta_iPET_100_test/yelp/p2-i2/results.txt: Score = 0.25, #Logits = 50000, #Labels = 5
    
        loglist = LogitsList(score=result_train, logits=logits)
        # print(f'loglist: {loglist}') # loglist: LogitsList(score=0.25, logits[:2]=[[5.260121, -2.7068188, -6.505483, 1.3350872, 19.978271], [3.1418624, 18.523603, 1.783086, 2.1958969, -1.2012017]])
        logits_lists[subdir] = loglist
        # print(f'logits_lists: {logits_lists}') #logits_lists: {'p2-i2': LogitsList(score=0.25, logits[:2]=[[5.260121, -2.7068188, -6.505483, 1.3350872, 19.978271], [3.1418624, 18.523603, 1.783086, 2.1958969, -1.2012017]]), 'p3-i0': LogitsList(score=0.31, logits[:2]=[[2.060473, -5.0971932, -3.5274274, -1.1608342, 18.032104], [3.3292046, 20.60177, 0.24962683, 4.426136, -3.725746]]), 'p3-i2': LogitsList(score=0.31, logits[:2]=[[2.7231414, -3.5293348, -4.3000526, 0.75283766, 19.091772], [2.283721, 22.406498, 5.0643315, 6.0301456, -4.6486263]]), 'p1-i1': LogitsList(score=0.38, logits[:2]=[[4.611694, -3.070371, -5.9071527, -1.001478, 18.717123], [3.0648386, 21.078388, 1.4702404, 4.721809, -3.744368]]), 'p1-i0': LogitsList(score=0.38, logits[:2]=[[4.2726197, -4.791045, -4.2288995, -4.120239, 16.28123], [7.0945616, 22.5028, -4.251675, 5.5282965, -2.2665176]]), 'p0-i2': LogitsList(score=0.41, logits[:2]=[[4.4773693, -3.9174242, -3.3116395, 3.76114, 20.80512], [2.6890187, 20.189182, -0.5262206, 3.0182457, -2.5321305]]), 'p0-i1': LogitsList(score=0.41, logits[:2]=[[5.4069896, -3.756437, -3.2553425, 3.62263, 20.259268], [1.9288068, 20.6073, 0.8252569, 3.2210388, -2.0031898]]), 'p2-i1': LogitsList(score=0.25, logits[:2]=[[4.488108, -3.852061, -5.194917, 4.3645945, 19.91591], [2.7579076, 18.91675, 3.458161, 4.735767, -1.0149256]]), 'p3-i1': LogitsList(score=0.31, logits[:2]=[[4.350354, -1.9898229, -3.1502078, 1.4578516, 18.866089], [4.459938, 20.462309, -0.3979785, 5.9158673, 1.2597402]]), 'p0-i0': LogitsList(score=0.41, logits[:2]=[[5.9478846, -2.246691, -4.30881, 2.3847644, 19.828459], [3.9836926, 21.255121, 7.752589, 4.2288046, -2.4681304]]), 'p2-i0': LogitsList(score=0.25, logits[:2]=[[5.4967856, -2.1803536, -4.707951, 3.189568, 20.856955], [4.4511113, 19.426277, 2.2305303, 4.4012446, -1.4902421]]), 'p1-i2': LogitsList(score=0.38, logits[:2]=[[5.837471, -3.7982347, -2.9907975, 3.5469222, 20.82018], [4.035328, 20.567448, 3.0972004, 3.0757291, -4.032412]])}
   
    
    for subdir in subdirs:
        other_logits_lists = [] # 假設當前subdir是p2-i2, 則取得除了p2-i2的其他pattern的LogitsList
        for subdir_name, logits_list in logits_lists.items():
            print(f'subdir_name: {subdir_name}')
            if subdir_name != subdir:
                other_logits_lists.append(logits_list)
        # other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        if subdir== 'p2-i2':
            print(f'other_logits_lists: {other_logits_lists}') # other_logits_lists: [LogitsList(score=0.31, logits[:2]=[[2.060473, -5.0971932, -3.5274274, -1.1608342, 18.032104], [3.3292046, 20.60177, 0.24962683, 4.426136, -3.725746]]), LogitsList(score=0.31, logits[:2]=[[2.7231414, -3.5293348, -4.3000526, 0.75283766, 19.091772], [2.283721, 22.406498, 5.0643315, 6.0301456, -4.6486263]]), LogitsList(score=0.38, logits[:2]=[[4.611694, -3.070371, -5.9071527, -1.001478, 18.717123], [3.0648386, 21.078388, 1.4702404, 4.721809, -3.744368]]), LogitsList(score=0.38, logits[:2]=[[4.2726197, -4.791045, -4.2288995, -4.120239, 16.28123], [7.0945616, 22.5028, -4.251675, 5.5282965, -2.2665176]]), LogitsList(score=0.41, logits[:2]=[[4.4773693, -3.9174242, -3.3116395, 3.76114, 20.80512], [2.6890187, 20.189182, -0.5262206, 3.0182457, -2.5321305]]), LogitsList(score=0.41, logits[:2]=[[5.4069896, -3.756437, -3.2553425, 3.62263, 20.259268], [1.9288068, 20.6073, 0.8252569, 3.2210388, -2.0031898]]), LogitsList(score=0.25, logits[:2]=[[4.488108, -3.852061, -5.194917, 4.3645945, 19.91591], [2.7579076, 18.91675, 3.458161, 4.735767, -1.0149256]]), LogitsList(score=0.31, logits[:2]=[[4.350354, -1.9898229, -3.1502078, 1.4578516, 18.866089], [4.459938, 20.462309, -0.3979785, 5.9158673, 1.2597402]]), LogitsList(score=0.41, logits[:2]=[[5.9478846, -2.246691, -4.30881, 2.3847644, 19.828459], [3.9836926, 21.255121, 7.752589, 4.2288046, -2.4681304]]), LogitsList(score=0.25, logits[:2]=[[5.4967856, -2.1803536, -4.707951, 3.189568, 20.856955], [4.4511113, 19.426277, 2.2305303, 4.4012446, -1.4902421]]), LogitsList(score=0.38, logits[:2]=[[5.837471, -3.7982347, -2.9907975, 3.5469222, 20.82018], [4.035328, 20.567448, 3.0972004, 3.0757291, -4.032412]])]

        
        subdir_train_set = generate_train_set(other_logits_lists,
                                              labels=labels,
                                              original_data=all_train_data,
                                              num_examples=args.num_examples,
                                              logits_percentage=args.logits_percentage,
                                              reduction=args.reduction,
                                              n_most_likely=args.n_most_likely)
        print(f'len(subdir_train_set): {len(subdir_train_set)}') # len(subdir_train_set): 100

        # save_examples(靜態方法)通過類名.靜態方法名()的形式呼叫
        InputExample.save_examples(
            subdir_train_set,
            os.path.join(args.output_dir, f'{subdir}-train.txt'),
        )

        p = os.path.join('./output_roberta_iPET_100_test/next-gen-train-sets','p2-i2-train.txt')

        additional_data = InputExample.load_examples(p)
        for example in additional_data:
            print(f'example: {example}')
        print(f'len(additional_data): {len(additional_data)}') #len(additional_data): 100
        



        
        
    


if __name__ == "__main__":
    main()
