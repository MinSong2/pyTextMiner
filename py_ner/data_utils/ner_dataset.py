from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Union
from torch import nn

from transformers import PreTrainedTokenizer
import codecs
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

from typing import Tuple, Callable, List # https://m.blog.naver.com/PostView.nhn?blogId=passion053&logNo=221070020739&proxyReferer=https%3A%2F%2Fwww.google.com%2F

import json
import re
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from py_ner.kobert.pytorch_kobert import get_pytorch_kobert_model
from py_ner.kobert.utils import get_tokenizer
from py_ner.data_utils.vocab_tokenizer import Vocabulary, Tokenizer
from py_ner.data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

class NamedEntityRecognitionDataset(Dataset):
    def __init__(self, train_data_dir: str, model_dir=Path('data')) -> None:
        """
        :param train_data_in:
        :param transform_fn:
        """

        list_of_total_source_no, list_of_total_source_str, list_of_total_target_str = [], [], []

        self.model_dir = model_dir
        if os.path.isdir(train_data_dir):
            print("\nIt is a directory")
            list_of_total_source_no, list_of_total_source_str, \
                list_of_total_target_str = self.load_data(train_data_dir=train_data_dir)
        elif os.path.isfile(train_data_dir):
            print("\nIt is a normal file")
            list_of_source_no, list_of_source_str, \
                list_of_target_str = self.load_data_from_txt(file_full_name=train_data_dir)
            list_of_total_source_no.extend(list_of_source_no)
            list_of_total_source_str.extend(list_of_source_str)
            list_of_total_target_str.extend(list_of_target_str)

        self.create_ner_dict(list_of_total_target_str)
        self._corpus = list_of_total_source_str
        self._label = list_of_total_target_str

    def set_transform_fn(self, transform_source_fn, transform_target_fn):
        self._transform_source_fn = transform_source_fn
        self._transform_target_fn = transform_target_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # preprocessing
        # str -> id -> cls, sep -> pad

        token_ids_with_cls_sep, tokens, prefix_sum_of_token_start_index = self._transform_source_fn(self._corpus[idx].lower())
        list_of_ner_ids, list_of_ner_label = self._transform_target_fn(self._label[idx], tokens, prefix_sum_of_token_start_index)

        x_input = torch.tensor(token_ids_with_cls_sep).long()
        token_type_ids = torch.tensor(len(x_input[0]) * [0])
        label = torch.tensor(list_of_ner_ids).long()
        # print("x_input.size(): ", x_input.size())
        # print("token_type_ids: ", token_type_ids)
        # print("label.size(): ", label.size())

        return x_input[0], token_type_ids, label

    def create_ner_dict(self, list_of_total_target_str):
        """ if you want to build new json file, you should delete old version. """

        if not os.path.exists(self.model_dir + "/ner_to_index.json"):
            regex_ner = re.compile('<(.+?):[A-Z]{3}>')
            list_of_ner_tag = []
            for label_text in list_of_total_target_str:
                regex_filter_res = regex_ner.finditer(label_text)
                for match_item in regex_filter_res:
                    ner_tag = match_item[0][-4:-1]
                    if ner_tag not in list_of_ner_tag:
                        list_of_ner_tag.append(ner_tag)

            ner_to_index = {"[CLS]":0, "[SEP]":1, "[PAD]":2, "[MASK]":3, "O": 4}
            for ner_tag in list_of_ner_tag:
                ner_to_index['B-'+ner_tag] = len(ner_to_index)
                ner_to_index['I-'+ner_tag] = len(ner_to_index)

            # save ner dict in data_in directory
            with open(self.model_dir / 'ner_to_index.json', 'w', encoding='utf-8') as io:
                json.dump(ner_to_index, io, ensure_ascii=False, indent=4)
            self.ner_to_index = ner_to_index
        else:
            self.set_ner_dict()

    def set_ner_dict(self):
        with open(self.model_dir + "/ner_to_index.json", 'rb') as f:
            self.ner_to_index = json.load(f)

    def load_data(self, train_data_dir):
        list_of_file_name = [file_name for file_name in os.listdir(train_data_dir) if '.txt' in file_name]
        list_of_full_file_path = [train_data_dir + "/" + file_name for file_name in list_of_file_name]
        print("num of files: ", len(list_of_full_file_path))

        list_of_total_source_no, list_of_total_source_str, list_of_total_target_str = [], [], []
        for i, full_file_path in enumerate(list_of_full_file_path):
            list_of_source_no, list_of_source_str, list_of_target_str = self.load_data_from_txt(file_full_name=full_file_path)
            list_of_total_source_str.extend(list_of_source_str)
            list_of_total_target_str.extend(list_of_target_str)
        assert len(list_of_total_source_str) == len(list_of_total_target_str)

        return list_of_total_source_no, list_of_total_source_str, list_of_total_target_str

    def load_data_from_txt(self, file_full_name):
        with codecs.open(file_full_name, "r", "utf-8") as io:
            lines = io.readlines()

            # parsing에 문제가 있어서 아래 3개 변수 도입!
            prev_line = ""
            save_flag = False
            count = 0
            sharp_lines = []

            for line in lines:
                if prev_line == "\n" or prev_line == "":
                    save_flag = True
                if line[:3] == "## " and save_flag is True:
                    count += 1
                    sharp_lines.append(line[3:])
                if count == 3:
                    count = 0
                    save_flag = False

                prev_line = line
            list_of_source_no, list_of_source_str, list_of_target_str = sharp_lines[0::3], sharp_lines[1::3], sharp_lines[2::3]
        return list_of_source_no, list_of_source_str, list_of_target_str

class NamedEntityRecognitionFormatter():
    """ NER formatter class """
    def __init__(self, vocab=None, tokenizer=None, maxlen=30, model_dir=Path('data_in')):

        if vocab is None or tokenizer is None:
            tok_path = get_tokenizer()
            self.ptr_tokenizer = SentencepieceTokenizer(tok_path)
            self.ptr_detokenizer = SentencepieceDetokenizer(tok_path)
            _, vocab_of_gluonnlp = get_pytorch_kobert_model()
            token2idx = vocab_of_gluonnlp.token_to_idx
            self.vocab = Vocabulary(token2idx=token2idx)
            self.tokenizer = Tokenizer(vocab=self.vocab, split_fn=self.ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=maxlen)
        else:
            self.vocab = vocab
            self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.model_dir = model_dir

    def transform_source_fn(self, text):
        # text = "첫 회를 시작으로 13일까지 4일간 총 4회에 걸쳐 매 회 2편씩 총 8편이 공개될 예정이다."
        # label_text = "첫 회를 시작으로 <13일:DAT>까지 <4일간:DUR> 총 <4회:NOH>에 걸쳐 매 회 <2편:NOH>씩 총 <8편:NOH>이 공개될 예정이다."
        # text = "트래버 모리슨 학장은 로스쿨 학생과 교직원이 바라라 전 검사의 사법정의에 대한 깊이 있는 지식과 경험으로부터 많은 것을 배울 수 있을 것이라고 말했다."
        # label_text = "<트래버 모리슨:PER> 학장은 로스쿨 학생과 교직원이 <바라라:PER> 전 검사의 사법정의에 대한 깊이 있는 지식과 경험으로부터 많은 것을 배울 수 있을 것이라고 말했다."
        tokens = self.tokenizer.split(text)
        token_ids_with_cls_sep = self.tokenizer.list_of_string_to_arr_of_cls_sep_pad_token_ids([text])

        # save token sequence length for matching entity label to sequence label
        prefix_sum_of_token_start_index = []
        sum = 0
        for i, token in enumerate(tokens):
            if i == 0:
                prefix_sum_of_token_start_index.append(0)
                sum += len(token) - 1
            else:
                prefix_sum_of_token_start_index.append(sum)
                sum += len(token)
        return token_ids_with_cls_sep, tokens, prefix_sum_of_token_start_index


    def transform_target_fn(self, label_text, tokens, prefix_sum_of_token_start_index):
        """
        인풋 토큰에 대응되는 index가 토큰화된 엔티티의 index 범위 내에 있는지 체크해서 list_of_ner_ids를 생성함
        이를 위해서 B 태그가 시작되었는지 아닌지도 체크해야함
        매칭하면 entity index를 증가시켜서 다음 엔티티에 대해서도 검사함
        :param label_text:
        :param tokens:
        :param prefix_sum_of_token_start_index:
        :return:
        """
        regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
        regex_filter_res = regex_ner.finditer(label_text)

        list_of_ner_tag = []
        list_of_ner_text = []
        list_of_tuple_ner_start_end = []

        count_of_match = 0
        for match_item in regex_filter_res:
            ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
            ner_text = match_item[1]  # <4일간:DUR> -> 4일간
            start_index = match_item.start() - 6 * count_of_match  # delete previous '<, :, 3 words tag name, >'
            end_index = match_item.end() - 6 - 6 * count_of_match

            list_of_ner_tag.append(ner_tag)
            list_of_ner_text.append(ner_text)
            list_of_tuple_ner_start_end.append((start_index, end_index))
            count_of_match += 1

        list_of_ner_label = []
        entity_index = 0
        is_entity_still_B = True
        for tup in zip(tokens, prefix_sum_of_token_start_index):
            token, index = tup

            if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
                index += 1  # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

            if entity_index < len(list_of_tuple_ner_start_end):
                start, end = list_of_tuple_ner_start_end[entity_index]

                if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                    is_entity_still_B = True
                    entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                    start, end = list_of_tuple_ner_start_end[entity_index]

                if start <= index and index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                    entity_tag = list_of_ner_tag[entity_index]
                    if is_entity_still_B is True:
                        entity_tag = 'B-' + entity_tag
                        list_of_ner_label.append(entity_tag)
                        is_entity_still_B = False
                    else:
                        entity_tag = 'I-' + entity_tag
                        list_of_ner_label.append(entity_tag)
                else:
                    is_entity_still_B = True
                    entity_tag = 'O'
                    list_of_ner_label.append(entity_tag)

            else:
                entity_tag = 'O'
                list_of_ner_label.append(entity_tag)

            # print((token, index, entity_tag), end=' ')

        with open(self.model_dir +  "/ner_to_index.json", 'rb') as f:
            self.ner_to_index = json.load(f)
        # ner_str -> ner_ids -> cls + ner_ids + sep -> cls + ner_ids + sep + pad + pad .. + pad
        list_of_ner_ids = [self.ner_to_index['[CLS]']] + [self.ner_to_index[ner_tag] for ner_tag in list_of_ner_label] + [self.ner_to_index['[SEP]']]
        list_of_ner_ids = self.tokenizer._pad([list_of_ner_ids], pad_id=self.vocab.PAD_ID, maxlen=self.maxlen)[0]

        return list_of_ner_ids, list_of_ner_label



@dataclass
class DataSample:
    """
    A single training/test example (sentence) for token classification.
    """
    words: List[str]
    labels: List[str]

@dataclass
class InputBert:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a BERT model.
    """
    input_ids: torch.tensor
    attention_mask: torch.tensor
    token_type_ids: torch.tensor
    labels: Optional[torch.tensor] = None

class NerDataset(Dataset):
    def __init__(self,
                 dataset: List[DataSample],
                 tokenizer: PreTrainedTokenizer,
                 labels2ind: Dict[str, int],
                 max_len_seq: int = 512,
                 bert_hugging: bool = True):
        """
        Class that builds a torch Dataset specially designed for NER data.
        Args:
            dataset (list of `DataSample` instances): Each data sample is a dataclass
                that contains two fields: `words` and `labels`. Both are lists of `str`.
            tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers
                library. Usually loaded as `AutoTokenizer.from_pretrained(...)`.
            labels2ind (`dict`): maps `str` class labels into `int` indexes.
            max_len_seq (`int`): Max length sequence for each example (sentence).
            bert_hugging (`bool`):
        """
        super(NerDataset).__init__()
        self.bert_hugging = bert_hugging
        self.max_len_seq = max_len_seq
        self.label2ind = labels2ind
        self.features = data2tensors(data=dataset,
                                     tokenizer=tokenizer,
                                     label2idx=self.label2ind,
                                     max_seq_len=max_len_seq,
                                     pad_token_label_id=nn.CrossEntropyLoss().ignore_index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Union[Dict[str, torch.tensor],
                                      Tuple[List[torch.tensor], torch.tensor]]:
        if self.bert_hugging:
            return asdict(self.features[i])
        else:
            inputs = asdict(self.features[i])
            labels = inputs.pop('labels')
            return list(inputs.values()), labels


def get_labels(data: List[DataSample]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Automatically extract labels types from the data and its count.
    Args:
        data (list of `DataSample`): Each data sample is a dataclass that contains
            two fields: `words` and `labels`. Both are lists of `str`.

    Returns:
        labels2idx (`dict`): maps `str` class labels into `int` indexes.
        labels_count(`dict`): The number of words for each class label that appears in
            the dataset. Usufull information if you want to apply class weights on
            imbalanced data.

    """
    labels = set()
    labels_counts = defaultdict(int)
    for sent in data:
        labels.update(sent.labels)

        for label_ in sent.labels:
            labels_counts[label_] += 1

    if "O" not in labels:
        labels.add('O')
        labels_counts['0'] = 0

    # Convert list of labels ind a mapping labels -> index
    labels2idx = {label_: i for i, label_ in enumerate(labels)}
    return labels2idx, dict(labels_counts)


def get_class_weight_tensor(labels2ind: Dict[str, int],
                            labels_count: Dict[str, int]) -> torch.Tensor:
    """
    Get the class weights based on the class labels frequency within the dataset.
    Args:
        labels2ind (`dict`): maps `str` class labels into `int` indexes.
        labels_count (`dict`): The number of words for each class label that appears in
            the dataset.

    Returns:
        torch.Tensor with the class weights. Size (num_classes).

    """
    label2ind_list = [(k, v) for k, v in labels2ind.items()]
    label2ind_list.sort(key=lambda x: x[1])
    total_labels = sum([count for label, count in labels_count.items()])
    class_weights = [total_labels/labels_count[label] for label, _ in label2ind_list]
    return torch.tensor(np.array(class_weights)/max(class_weights), dtype=torch.float32)


def read_data_from_file(file_path: str, sep: str = '\t') -> List[DataSample]:
    """
    Load data from a txt file (BIO tagging format) and transform it into the
    required format (list of `DataSample` instances).
    Args:
        file_path (`str`): complete path where the data is located (path + filename).
        sep (`str`): Symbol used to separete word from label at each line. Default `\t`.

    Returns:
        List of `DataSample` instances containing words and labels.

    """
    examples = []
    words = []
    labels = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            splits = line.split(sep)
            if len(splits) > 1:
                words.append(splits[0])
                labels.append(splits[-1].replace('\n', ''))
            else:
                examples.append(DataSample(words=words, labels=labels))
                words = []
                labels = []
    return examples


def data2tensors(data: List[DataSample],
                 tokenizer: PreTrainedTokenizer,
                 label2idx: Dict[str, int],
                 pad_token_label_id: int = -100,
                 max_seq_len: int = 512) -> List[InputBert]:
    """
    Takes data and converts it into tensors to feed the neural network.
    Args:
        data (`list`): List of `DataSample` instances containing words and labels.
        tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers
            library. Usually loaded as `AutoTokenizer.from_pretrained(...)`.
        label2idx (`dict`): maps `str` class labels into `int` indexes.
        pad_token_label_id (`int`): index to define the special token [PAD]
        max_seq_len (`int`): Max sequence length.

    Returns:
        List of `InputBert` instances. `InputBert` is a dataclass that contains
        `input_ids`, `attention_mask`, `token_type_ids` and `labels` (Optional).

    """

    features = []
    for sentence in data:
        tokens = []
        label_ids = []
        for word, label in zip(sentence.words, sentence.labels):
            subword_tokens = tokenizer.tokenize(text=word)

            # BERT could return an empty list of subtokens
            if len(subword_tokens) > 0:
                tokens.extend(subword_tokens)

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label2idx[label]] + [pad_token_label_id] * (len(subword_tokens) - 1))
                # if label.startswith('B'):
                #     label_ids.extend([label2idx[label]] + [label2idx[f"I{label[1:]}"]] * (len(subword_tokens) - 1))
                # else:
                #     label_ids.extend([label2idx[label]] + [label2idx[label]] * (len(subword_tokens) - 1))

        # Drop part of the sequence longer than max_seq_len (account also for [CLS] and [SEP])
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
            label_ids = label_ids[: max_seq_len - 2]

        # Add special tokens  for the list of tokens and its corresponding labels.
        # For BERT: cls_token = '[CLS]' and sep_token = '[SEP]'
        # For RoBERTa: cls_token = '<s>' and sep_token = '</s>'
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]

        # Create an attention mask (used to locate the padding)
        padding_len = (max_seq_len - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * padding_len

        # Add padding
        tokens += [tokenizer.pad_token] * padding_len
        label_ids += [pad_token_label_id] * padding_len

        # Convert tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create segment_id. All zeros since we only have one sentence
        segment_ids = [0] * max_seq_len

        # Assert all the input has the expected length
        assert len(input_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        # Append input features for each sequence/sentence
        features.append((InputBert(input_ids=torch.tensor(input_ids),
                                   attention_mask=torch.tensor(attention_mask),
                                   token_type_ids=torch.tensor(segment_ids),
                                   labels=torch.tensor(label_ids))))
    return features

if __name__ == '__main__':
    text = "첫 회를 시작으로 13일까지 4일간 총 4회에 걸쳐 매 회 2편씩 총 8편이 공개될 예정이다."
    label_text = "첫 회를 시작으로 <13일:DAT>까지 <4일간:DUR> 총 <4회:NOH>에 걸쳐 매 회 <2편:NOH>씩 총 <8편:NOH>이 공개될 예정이다."
    ner_formatter = NamedEntityRecognitionFormatter()
    token_ids_with_cls_sep, tokens, prefix_sum_of_token_start_index = ner_formatter.transform_source_fn(text)
    ner_formatter.transform_target_fn(label_text, tokens, prefix_sum_of_token_start_index)


