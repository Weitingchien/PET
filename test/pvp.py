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
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import re
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
import torch.nn.functional as F
from transformers import InputExample, PreTrainedTokenizer, GPT2Tokenizer

import logging

logger = logging.getLogger()


def _prepare(word: str, tokenizer: PreTrainedTokenizer) -> str:
    """ 
        如果給定的tokenizer是GPT2Tokenizer,那麼對給定的單詞進行tokenization,並確保結果只包含一個token。如果不是,就raise一個錯誤。
        如果給定的tokenizer不是GPT2Tokenizer,那麼直接返回原始的單詞。
    """
    if isinstance(tokenizer, GPT2Tokenizer):
        tokenized_word = tokenizer.tokenize(word, add_prefix_space=True)
        if len(tokenized_word) != 1:
            raise ValueError('"{}" is not a single-token word (tokenized: {})'.format(word, tokenized_word))
        return tokenized_word[0]
    return word


class PVP(ABC):

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_file: str = None):
        self.wrapper = wrapper
        self.pattern_id = pattern_id

        self.first_encode_call = True 

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            print(f'verbalizers: {verbalizers}')
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer = _prepare(verbalizer, self.wrapper.tokenizer)
                verbalizer_id = self.wrapper.tokenizer.convert_tokens_to_ids(verbalizer)
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self):
        return self.wrapper.tokenizer.mask_token

    @property
    def max_num_verbalizers(self):
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        return s, True

    def encode(self, example: InputExample) -> Tuple[List[int], List[int]]:
        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
        # build_inputs_with_special_tokens合併兩個序列,額外添加特殊token([CLS],[SEP])
        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        # 指示每個token屬於原始輸入中的哪一部分(tokens_a或tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        decoded_text = self.wrapper.tokenizer.decode(input_ids)
        # 解碼input_ids
        if self.first_encode_call:
            # print("Decoded text:", decoded_text)
            self.first_encode_call = False  # 確保只打印第一次調用的decoded_text

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_added_tokens(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> Tuple[
        List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]:
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        mask_token_id = self.wrapper.tokenizer.mask_token_id
        label_idx = input_ids.index(mask_token_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1 #將遮蔽位置的值設為1,其餘為-1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor, data_dict: dict=None, iteration: int=None) -> torch.Tensor:
        # mlm_labels: 表示哪些位置是被遮蔽的,shape為(batch_size, sequence_length)
        # logits: MLM任務的輸出(batch_size, sequence_length, vocab_size)
        # masked_logits包含了模型對輸入序列中被遮蔽位置的預測結果。它的row對應一個被遮蔽的位置,每一個column對應一個單詞在詞彙表中的位置。masked_logits的元素值表示模型認為該遮蔽位置填入對應單詞的"可能性"或"置信度"
        # print(f'(convert_mlm_logits_to_cls_logits): {logits}, shape: {logits.shape}')
        # logits: torch.Size([1, 256, 50265])
        # print(f'mlm_labels: {mlm_labels}')
        # print(f'logits: {logits.shape}') 
        """
            mlm_labels(針對yelp第一種pattern): tensor([[-1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1]], device='cuda:0')
        """
        print(f'logits.shape: {logits.shape}')
        # 提取被遮敝位置的預測分數(忽略其他位置的預測分數)
        masked_logits = logits[mlm_labels >= 0]
        # 找到被遮蔽位置的索引
        masked_indices = (mlm_labels >= 0).nonzero(as_tuple=True)[0]
        # print(f'masked_logits: {masked_logits}, The length of the sequence is: {masked_logits.shape[1]}, shape: {masked_logits.shape}')
        """
            masked_logits: tensor([[-4.7968, -4.4525, -4.7972,  ..., -4.0598, -4.4333, -2.0190]],
            device='cuda:0'), The length of the sequence is: 30522, shape: torch.Size([1, 30522])
        """
        if data_dict is None:
            cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(l) for l in masked_logits])

        # 對每個被遮罩位置的預測分數進行轉換
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(l, data_dict, iteration) for l in masked_logits])
        """
            torch.stack是PyTorch中的一個函數,用於將多個張量(tensor)沿著一個新的維度進行拼接 (默認為0,即在第一個維度上進行拼接)
        """
        print(f'cls_logits: {cls_logits}, The length of the sequence is: {cls_logits.shape[1]}, shape: {cls_logits.shape}')
        """
            cls_logits: tensor([[3.8716, 3.3936, 5.0386, 6.4018, 7.7815]], device='cuda:0') len(cls_logits): 5, shape: torch.Size([1, 5])
        """

        # 找到每個被遮蔽位置預測分數最高的單字
        predicted_words = []
        for logits in masked_logits:
            predicted_index = torch.argmax(logits).item()
            predicted_word = self.wrapper.tokenizer.convert_ids_to_tokens(predicted_index)
            predicted_words.append(predicted_word)

        return cls_logits, predicted_words

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor, data_dict: dict=None, iteration: int=None) -> torch.Tensor:
        # print(f'(_convert_single_mlm_logits_to_cls_logits) logits: {logits}, shape: {logits.shape}')
        # logits: tensor([[-4.7968, -4.4525, -4.7972,  ..., -4.0598, -4.4333, -2.0190]], shape: torch.Size([30522])

        # 將MLM的token id映射到分類任務的label id
        # 儲存了將MLM token id映射到分類標籤id的映射關係
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        print(f'm2c: {m2c}')
        """
            tensor([[6587], [1099], [8578], [205], [372]], device='cuda:0')
            m2c: tensor([[ 6587, 11385], [ 1099,    -1], [ 8578,    -1], [  205,    -1], [  372,    -1]], device='cuda:0')
        """
         # 將logits和m2c存儲到data_dict中
        if data_dict is not None:
            # 將logits和m2c存儲到data_dict中
            key = f'p{self.pattern_id}-i{iteration}'
            data_dict[key] = {'logits': logits.cpu(), 'm2c': m2c.cpu()}


        tokenizer = self.wrapper.tokenizer
        """
        for row in m2c:
            token_id = row.item()  # 獲取token id
            token = tokenizer.convert_ids_to_tokens(token_id)  # 將id轉換為token
            print(f"Token id: {token_id}, Token: {token}")
            """
        """
                Token id: 6587, Token: Ġterrible
                Token id: 1099, Token: Ġbad
                Token id: 8578, Token: Ġokay
                Token id: 205, Token: Ġgood
                Token id: 372, Token: Ġgreat
            """

        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)
        print(f'filler_len: {filler_len}, shape: {filler_len.shape}')
        """
            filler_len: tensor([1., 1., 1., 1., 1.], device='cuda:0'), shape: torch.Size([5])
            filler_len: tensor([2., 1., 1., 1., 1.], device='cuda:0'), shape: torch.Size([5])
        """

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        """
            (Claude 3 Opus):
            torch.zeros_like(m2c)創建了一個與m2c形狀相同, 但所有元素都為0的新tensor
            對於m2c中每個元素(元素值對應模型預測token的機率, 30522個單字),如果它大於0(即有效的映射id),那麼就選取該元素本身的值。
            如果m2c中的元素為0或負數(無效映射),則選取torch.zeros_like(m2c)對應位置上的0值。
        """
        # 選擇每個標籤對應的最大 verbalizer(預測分數最高的verbalizer) 的預測分數
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        """
            整行代碼的作用是:從logits中選擇那些在m2c中有對應正數值(即有效的標籤-verbalizer映射)的元素
        """
        print(f'(_convert_single_mlm_logits_to_cls_logits-1): cls_logits: {cls_logits}')
        """
            (_convert_single_mlm_logits_to_cls_logits-1): cls_logits: tensor([[2.4382],
                [3.4054], [5.3409], [5.8134],  [5.6926]], device='cuda:0')
            (_convert_single_mlm_logits_to_cls_logits-1): cls_logits: tensor([[ 8.3138,  8.0406],
                [ 8.3874, -2.1249], [ 7.2761, -2.1249], [ 8.3397, -2.1249], [ 7.5939, -2.1249]],
                device='cuda:0', grad_fn=<IndexBackward>)
        """
        # 過濾掉無效的預測分數
        cls_logits = cls_logits * (m2c > 0).float()
        # 保險措施,確保即使在某些標籤沒有有效verbalizer的情況下,我們也能得到正確的分類分數
        print(f'(_convert_single_mlm_logits_to_cls_logits-2): cls_logits: {cls_logits}')
        """
            (_convert_single_mlm_logits_to_cls_logits-2): cls_logits: tensor([[ 9.3661],[ 7.0749],[ 8.3701],[ 9.2982],[12.0686]], device='cuda:0', grad_fn=<MulBackward0>)
            (_convert_single_mlm_logits_to_cls_logits-2): cls_logits: tensor([[8.3138, 8.0406],
                [8.3874, -0.0000], [7.2761, -0.0000], [8.3397, -0.0000],[7.5939, -0.0000]],
                device='cuda:0', grad_fn=<MulBackward0>)
        """

        # cls_logits.shape() == num_labels
        # 對選擇出的預測分數進行正規化
        cls_logits = cls_logits.sum(axis=1) / filler_len
        print(f'(_convert_single_mlm_logits_to_cls_logits-3) {cls_logits}')
        """
            (_convert_single_mlm_logits_to_cls_logits-3) tensor([ 9.3661,  7.0749,  8.3701,  9.2982, 12.0686], device='cuda:0',grad_fn=<DivBackward0>)
            (_convert_single_mlm_logits_to_cls_logits-3) tensor([8.1772, 8.3874, 7.2761, 8.3397, 7.5939],
                device='cuda:0',grad_fn=<DivBackward0>)
        """

        # print(f'cls_logits.shape: {cls_logits.shape}') # cls_logits.shape: torch.Size([5])
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class AgnewsPVP(PVP):
    VERBALIZER = {
        "1": ["World"],
        "2": ["Sports"],
        "3": ["Business"],
        "4": ["Tech"]
    }

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AgnewsPVP.VERBALIZER[label]


class YahooPVP(PVP):
    VERBALIZER = {
        "1": ["Society"],
        "2": ["Science"],
        "3": ["Health"],
        "4": ["Education"],
        "5": ["Computer"],
        "6": ["Sports"],
        "7": ["Business"],
        "8": ["Entertainment"],
        "9": ["Relationship"],
        "10": ["Politics"],
    }

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YahooPVP.VERBALIZER[label]


class MnliPVP(PVP):
    VERBALIZER_A = {
        "contradiction": ["Wrong"],
        "entailment": ["Right"],
        "neutral": ["Maybe"]
    }
    VERBALIZER_B = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> Tuple[
        List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]:

        text_a = self.shortenable(example.text_a.rstrip(string.punctuation))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return MnliPVP.VERBALIZER_A[label]
        return MnliPVP.VERBALIZER_B[label]


class YelpPolarityPVP(PVP):
    VERBALIZER = {
        "1": ["bad"],
        "2": ["good"]
    }

    def get_parts(self, example: InputExample) -> Tuple[
        List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]:

        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        elif self.pattern_id == 3:
            return [text], ['In summary, the restaurant is', self.mask, '.']
        else:
            raise ValueError(f"No pattern implemented for id {self.pattern_id}")

    def verbalize(self, label) -> List[str]:
        return YelpPolarityPVP.VERBALIZER[label]


class YelpFullPVP(YelpPolarityPVP):
    # VERBALIZER: 將標籤映射到詞彙表的單字
    VERBALIZER = {
        "1": ["terrible", "horrible"],
        "2": ["bad"],
        "3": ["okay"],
        "4": ["good"],
        "5": ["great"]
    }

    def verbalize(self, label) -> List[str]:
        return YelpFullPVP.VERBALIZER[label]


class XStancePVP(PVP):
    VERBALIZERS = {
        'en': {"FAVOR": ["Yes"], "AGAINST": ["No"]},
        'de': {"FAVOR": ["Ja"], "AGAINST": ["Nein"]},
        'fr': {"FAVOR": ["Oui"], "AGAINST": ["Non"]}
    }

    def get_parts(self, example: InputExample):

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return ['"', text_a, '"'], [self.mask, '. "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3 or self.pattern_id == 5:
            return [text_a], [self.mask, '.', text_b]

    def verbalize(self, label) -> List[str]:
        lang = 'de' if self.pattern_id < 2 else 'en' if self.pattern_id < 4 else 'fr'
        return XStancePVP.VERBALIZERS[lang][label]
