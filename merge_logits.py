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
This script can be used to merge logits obtained from different PVPs in order
to train a final model.
"""
import argparse
import ast
import os
from typing import List
import numpy as np

import log
from utils import LogitsList

logger = log.get_logger('root')


def merge_logits_lists(logits_lists: List[LogitsList], reduction: str = 'mean') -> LogitsList:
    """
        # p0-i0 'train_set_before_training': 0.41
            [5.9478846, -2.246691, -4.30881, 2.3847644, 19.828459],
        # p0-i1 'train_set_before_training': 0.41
            [5.4069896, -3.756437, -3.2553425, 3.62263, 20.259268], 
        # p0-i2 'train_set_before_training': 0.41,s
            [4.4773693, -3.9174242, -3.3116395, 3.76114, 20.80512],
        # p1-i0 'train_set_before_training': 0.38
            [4.2726197, -4.791045, -4.2288995, -4.120239, 16.28123],
        # p1-i1 'train_set_before_training': 0.38
            [4.611694, -3.070371, -5.9071527, -1.001478, 18.717123],
        # p1-i2 'train_set_before_training': 0.38
            [5.837471, -3.7982347, -2.9907975, 3.5469222, 20.82018],
        # p2-i0 'train_set_before_training': 0.25
            [5.4967856, -2.1803536, -4.707951, 3.189568, 20.856955],
        # p2-i1 'train_set_before_training': 0.25
            [4.488108, -3.852061, -5.194917, 4.3645945, 19.91591],
        # p2-i2 'train_set_before_training': 0.25
            [5.260121, -2.7068188, -6.505483, 1.3350872, 19.978271],
        # p3-i0 # 'train_set_before_training': 0.31
            [2.060473, -5.0971932, -3.5274274, -1.1608342, 18.032104],
        # p3-i1 'train_set_before_training': 0.31
            [4.350354, -1.9898229, -3.1502078, 1.4578516, 18.866089],
        # p3-i2 'train_set_before_training': 0.31
            [2.7231414, -3.5293348, -4.3000526, 0.75283766, 19.091772]
    """
    print(f'(merge_logits_lists): {len(logits_lists)}') # (merge_logits_lists): 12
    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    # logits: 把所有pattern的logits合併
    logits = np.array([ll.logits for ll in logits_lists])
    print(f'合併後: (merge_logits_lists): {logits} len(logits): {len(logits)}')
    weights = np.array([ll.score for ll in logits_lists])
    print(f'(weights): {weights}')

    if reduction == 'mean':
        logits = np.mean(logits, axis=0).tolist()
        print(f'算術平均後: {logits[:2]}, (merge_logits_lists) len(logits): {len(logits)}') # 50000
        # 算術平均後: [[4.577750933333334, -3.4113156, -4.282390041666667, 1.5110703633333333, 19.454373416666666], [3.601665933333333, 20.544787166666666, 1.729595685833333, 4.291581758333334, -2.3223124833333335]], (merge_logits_lists) len(logits): 50000
        # 對column方向加總後平均, 不同pattern對應的類別的分數
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    return LogitsList(score=-1, logits=logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="The dir in which the results of all PVPs are stored in separate subdirs. "
                             "Each subdir is expected to have a file 'results.txt' and a file 'logits.txt' in "
                             "it as created by 'run.py'")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The file where the merged logits are to be saved.")
    parser.add_argument("--overwrite_output_file", action='store_true',
                        help="Whether to overwrite the output file if it already exists")
    # action='store_true'當--overwrite_output_file出現在命令行中時,args.overwrite_output_file的值將被設置為True,否則它將是False
    parser.add_argument("--reduction", required=True, choices=['mean', 'wmean'],
                        help="The reduction strategy for merging logits. Must be one of 'mean' or 'wmean', "
                             "where the latter is short for 'weighted mean' and the weights for each PVP are "
                             "proportional to its score on the training set before fine-tuning.")
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        logger.error("Output file already exists")
        exit()

    subdirs = next(os.walk(args.logits_dir))[1]
    logger.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))
    # merge_logits - Found the following 12 subdirectories: ['p2-i2', 'p3-i0', 'p3-i2', 'p1-i1', 'p1-i0', 'p0-i2', 'p0-i1', 'p2-i1', 'p3-i1', 'p0-i0', 'p2-i0', 'p1-i2']

    all_logits_lists = []

    for subdir in subdirs:
        results_file = os.path.join(args.logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(args.logits_dir, subdir, 'logits.txt')
        logits = []

        # 取得所有pattern的train_set_before_training
        with open(results_file, 'r') as fh:
            results = ast.literal_eval(fh.read())
            # results: {'train_set_before_training': 0.25, 'train_set_after_training': 1.0, 'test_set_after_training': 0.5491}
            result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logger.info(
            f"File {results_file}: Score = {result_train}, #Logits = {len(logits)}, #Labels = {len(logits[0])}"
        )

        loglist = LogitsList(score=result_train, logits=logits)
        print(f'loglist: {loglist}')
        all_logits_lists.append(loglist)

    len(f'len(all_logits_lists): {len(all_logits_lists)}')

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=args.reduction)
    print(f'merged_loglist: {merged_loglist}')
    merged_loglist.save(args.output_file)


if __name__ == "__main__":
    main()
