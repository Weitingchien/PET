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
This script can be used to train and evaluate either a regular supervised model or a PET model on
one of the supported tasks and datasets.
"""
import os
import time
import csv
import log
import torch
import argparse
import statistics

from tasks import PROCESSORS, load_examples
from collections import defaultdict, Counter
from wrapper import TransformerModelWrapper, WRAPPER_TYPES
from utils import set_seed, eq_div, save_logits, LogitsList, InputExample

logger = log.get_logger('root')



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_examples", required=True, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--wrapper_type", required=True, choices=WRAPPER_TYPES,
                        help="The wrapper type - either sequence_classifier (corresponding to"
                             "regular supervised training) or mlm (corresponding to PET training)")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="The model type (currently supported are bert and roberta)")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(PROCESSORS.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Optional parameters
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--lm_train_examples_per_label", default=10000, type=int,
                        help="The total number of training examples for auxiliary language modeling, "
                             "where -1 equals all examples")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET training)")
    parser.add_argument("--repetitions", default=3, type=int,
                        help="The number of times to repeat training and testing with different seeds.")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET training)")
    parser.add_argument("--save_train_logits", action='store_true',
                        help="Whether to save logits on the lm_train_examples in a separate file. This takes some "
                             "additional time but is required for combining PVPs  (only for PET training)")
    parser.add_argument("--additional_data_dir", default=None, type=str,
                        help="Path to a directory containing additional automatically labeled training examples (only "
                             "for iPET)")
    parser.add_argument("--per_gpu_helper_batch_size", default=4, type=int,
                        help="Batch size for the auxiliary task (only for PET training)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary task (only for PET training)")
    parser.add_argument("--temperature", default=1, type=float,
                        help="Temperature used for combining PVPs (only for PET training)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET training)")
    parser.add_argument("--logits_file", type=str,
                        help="The logits file for combining multiple PVPs, which can be created using the"
                             "merge_logits.py script")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to perform lower casing")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    for i in range(10):
        print(f'device: {device}')
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError(f"Task '{args.task_name}' not found")
    processor = PROCESSORS[args.task_name]()
    args.output_mode = "classification"
    args.label_list = processor.get_labels()

    """
        The logits file for combining multiple PVPs, which can be created using the merge_logits.py script
    """
    args.use_logits = args.logits_file is not None 

    wrapper = None

    logger.info(f"Training/evaluation parameters: {args}")
    results = defaultdict(list)
    # ed_div根據label的數量等分數據
    train_examples_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
    print(f'train_examples_per_label: {train_examples_per_label}')
    print(f'args.train_examples: {args.train_examples} len(args.label_list): {len(args.label_list)}')
    test_examples_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
    print(f'args.test_examples: {args.test_examples} len(args.label_list): {len(args.label_list)}')
    print(f'test_examples_per_label: {test_examples_per_label}') # -1

    train_data = load_examples(args.task_name, args.data_dir, train_examples_per_label, evaluate=False)
    """
    for i, example in enumerate(train_data):
        print(f"Example {i+1}:")
        print(f"  Guid: {example.guid}")
        print(f"  Text A: {example.text_a}")
        print(f"  Text B: {example.text_b}")
        print(f"  Label: {example.label}")
        print(f"  Logits: {example.logits}")
        print()
    """

    labels = [example.label for example in train_data]
    label_counts = Counter(labels)
    print(f"Label counts in train_data: {label_counts}")
    # Label counts in train_data: Counter({'5': 20, '2': 20, '4': 20, '1': 20, '3': 20})
    


    eval_data = load_examples(args.task_name, args.data_dir, test_examples_per_label, evaluate=True)

    labels = [example.label for example in eval_data]
    label_counts = Counter(labels)
    print(f"Label counts in eval_data: {label_counts}")
    # Label counts in eval_data: Counter({'1': 10000, '3': 10000, '2': 10000, '4': 10000, '5': 10000})

    if train_data:
        print("First training example:")
        print(train_data[0])
        """
        First training example:
        {
            "guid": "train-0",
            "label": "5",
            "logits": null,
            "text_a": "dr. goldberg offers everything i look for in a general practitioner.  he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first.  really, what more do you need?  i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank.",
            "text_b": null
        }
        """
       
        print(f"Size of train data: {len(train_data)}") # Size of train data: 100
    


    if eval_data:
        """
        print("First evaluation example:")
        print(eval_data[0])
        
        for attribute, value in eval_data[0].__dict__.items():
            print(f"{attribute}: {value}")
        """
        print(f"Size of eval data: {len(eval_data)}") # Size of eval data: 50000


    if args.lm_training or args.save_train_logits or args.use_logits:
        # print(f'args.lm_train_examples_per_label: {args.lm_train_examples_per_label}')
        # Size of all_train_data: 10000 * 5 = 50000(總共有5個標籤,每個標籤各一萬筆訓練樣本)
        all_train_data = load_examples(args.task_name, args.data_dir, args.lm_train_examples_per_label, evaluate=False)
        print(f'all_train_data[0]: {all_train_data[0]}')
        
        # print("First few examples from all_train_data:")
        """
        for i in range(5):  # 打印前5個樣本
            print(f"Example {i+1}:")
            print(f"  Text: {all_train_data[i].text_a}")
            print(f"  Label: {all_train_data[i].label}")
        """

    else:
        all_train_data = None
    # print(f'Size of all_train_data: {len(all_train_data)}') # Size of all_train_data: 50000


    if args.use_logits:
        logits = LogitsList.load(args.logits_file).logits # 載入預先計算好的 logits 值
        assert len(logits) == len(all_train_data) # 檢查載入的 logits 的數量是否與訓練數據集 all_train_data 中的樣本數量相同
        logger.info(f"Got {len(logits)} logits from file {args.logits_file}")
        for example, example_logits in zip(all_train_data, logits):
            example.logits = example_logits
        """
        for example, example_logits in zip(all_train_data, logits):
            example.logits = example_logits
            print(f'Guid: {example.guid}')
            print(f"Text: {example.text_a}")
            print(f"Label: {example.label}")
            print(f"Logits: {example.logits}")
            print("---")
        """

    total_train_time = 0
    total_eval_time = 0

    for pattern_id in args.pattern_ids:
        args.pattern_id = pattern_id
        for iteration in range(args.repetitions):

            results_dict = {}

            output_dir = f"{args.output_dir}/p{args.pattern_id}-i{iteration}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if args.do_train:
                wrapper = TransformerModelWrapper(args)
                wrapper.model.to(device)


                results_dict['train_set_before_training'] = wrapper.eval(train_data, device, **vars(args))['acc']

                pattern_iter_train_data = []
                pattern_iter_train_data.extend(train_data) # len(task_train_data): 100
                # (iPET)
                if args.additional_data_dir:
                    p = os.path.join(
                        args.additional_data_dir,
                        f'p{args.pattern_id}-i{iteration}-train.txt',
                    )
                    additional_data = InputExample.load_examples(p)
                    for example in additional_data:
                        example.logits = None
                    pattern_iter_train_data.extend(additional_data)
                    logger.info(
                        f"Loaded {len(additional_data)} additional examples from {p}, total training size is now {len(pattern_iter_train_data)}"
                    )
                """
                    additional_data_dir: 添加400個訓練樣本(不包含當前pattern的模型, 綜合其他3個模型預測的標籤的訓練樣本)
                """
                print(f'len(pattern_iter_train_data): {len(pattern_iter_train_data)}') # len(pattern_iter_train_data): 500
                print(f'pattern_iter_train_data[0]: {pattern_iter_train_data[0]}')
                """
                pattern_iter_train_data[0]: {
                    "guid": "train-0",
                    "label": "5",
                    "logits": null,
                    "text_a": "dr. goldberg offers everything i look for in a general practitioner.  he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first.  really, what more do you need?  i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank.",
                    "text_b": null
                }
                """

        
        

                logger.info("Starting training...")
                start_time = time.time()

                data_dict, global_step, tr_loss = wrapper.train(
                    pattern_iter_train_data, device, iteration, 
                    helper_train_data=all_train_data if args.lm_training or args.use_logits else None,
                    tmp_dir=output_dir, **vars(args))


                train_time = time.time() - start_time
                total_train_time += train_time
                train_minutes, train_seconds = divmod(train_time, 60)
                logger.info(f"Training time for pattern {pattern_id}, iteration {iteration}: {train_minutes:.0f}m {train_seconds:.0f}s")


                # 在這裡創建訓練數據的映射CSV文件
                """
                for folder, values in data_dict.items():
                    print(f'folder: {folder}, values: {values}')
                    # create_mapping_csv(values['logits'], values['m2c'], wrapper.tokenizer, output_dir, 'mapping_train.csv')
                """

                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
                logger.info("Training complete")
    
                results_dict['train_set_after_training'] = wrapper.eval(train_data, device, **vars(args))['acc']


                with open(os.path.join(output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                logger.info(f"Saving trained model at {output_dir}...")
                wrapper.save(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving complete")
            
                # only for PET training
                if args.save_train_logits:
                    logits = wrapper.eval(all_train_data, device, output_logits=True, **vars(args))
                    save_logits(os.path.join(output_dir, 'logits.txt'), logits)

                if not args.do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()
            
            # Evaluation
            if args.do_eval:
                logger.info("Starting evaluation...")
                if not wrapper:
                    wrapper = TransformerModelWrapper.from_pretrained(output_dir)
                    wrapper.model.to(device)

                start_time = time.time()

                result = wrapper.eval(eval_data, device, **vars(args))

                eval_time = time.time() - start_time
                total_eval_time += eval_time
                eval_minutes, eval_seconds = divmod(eval_time, 60)
                logger.info(f"Evaluation time for pattern {pattern_id}, iteration {iteration}: {eval_minutes:.0f}m {eval_seconds:.0f}s")



                logger.info(f"--- RESULT (pattern_id={pattern_id}, iteration={iteration}) ---")
                logger.info(result)

                results_dict['test_set_after_training'] = result['acc']


                with open(os.path.join(output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                for key, value in result.items():
                    results[f'{key}-p{args.pattern_id}'].append(value)

                wrapper.model = None
                torch.cuda.empty_cache()


    # 計算平均訓練時間和評估時間
    avg_train_time = total_train_time / (len(args.pattern_ids) * args.repetitions)
    avg_eval_time = total_eval_time / (len(args.pattern_ids) * args.repetitions)

    avg_train_minutes, avg_train_seconds = divmod(avg_train_time, 60)
    avg_eval_minutes, avg_eval_seconds = divmod(avg_eval_time, 60)

    logger.info(f"Average training time: {avg_train_minutes:.0f}m {avg_train_seconds:.0f}s")
    logger.info(f"Average evaluation time: {avg_eval_minutes:.0f}m {avg_eval_seconds:.0f}s")


    # 將平均訓練時間和評估時間寫入檔案
    with open(os.path.join(args.output_dir, "exp_runtime_logs.txt"), "w") as f:
        f.write(f"Average training time: {avg_train_minutes:.0f}m {avg_train_seconds:.0f}s\n")
        f.write(f"Average evaluation time: {avg_eval_minutes:.0f}m {avg_eval_seconds:.0f}s\n")

    logger.info("=== OVERALL RESULTS ===")

    with open(os.path.join(args.output_dir, 'result_test.txt'), 'w') as fh:
        for key, values in results.items():
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            result_str = f"{key}: {mean} +- {stdev}"
            logger.info(result_str)
            fh.write(result_str + '\n')

        all_results = [result for pattern_results in results.values() for result in pattern_results]
        all_mean = statistics.mean(all_results)
        # 標準差: 用於衡量每個數值與平均值之間的偏差
        all_stdev = statistics.stdev(all_results)
        result_str = f"acc-all-p: {all_mean} +- {all_stdev}"
        logger.info(result_str)
        fh.write(result_str + '\n')
        

if __name__ == "__main__":
    main()
