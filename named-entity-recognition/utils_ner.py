# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import pdb
import json
import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock

from tqdm import tqdm
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    bias_tensor: Optional[List[int]] = None
    data_type: Optional[List[int]] = None

class Split(Enum):
    train = "train"
    dev = "devel"
    test = "test"

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class NerDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            is_pmi=False,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    examples = read_examples_from_file(data_dir, mode)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                        data_name=data_dir.split('/')[-1],
                        data_type=mode.value,
                        is_pmi=is_pmi,
                    )
                    # logger.info(f"Saving features into cached file {cached_features_file}")
                    # torch.save(self.features, cached_features_file)
                
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFNerDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = -1
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            examples = read_examples_from_file(data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(tokenizer.padding_side == "left"),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )

            def gen():
                for ex in self.features:
                    if ex.token_type_ids is None:
                        yield (
                            {"input_ids": ex.input_ids, "attention_mask": ex.attention_mask},
                            ex.label_ids,
                        )
                    else:
                        yield (
                            {
                                "input_ids": ex.input_ids,
                                "attention_mask": ex.attention_mask,
                                "token_type_ids": ex.token_type_ids,
                            },
                            ex.label_ids,
                        )

            if "token_type_ids" not in tokenizer.model_input_names:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                    (
                        {"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])},
                        tf.TensorShape([None]),
                    ),
                )
            else:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                    (
                        {
                            "input_ids": tf.TensorShape([None]),
                            "attention_mask": tf.TensorShape([None]),
                            "token_type_ids": tf.TensorShape([None]),
                        },
                        tf.TensorShape([None]),
                    ),
                )

        def get_dataset(self):
            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    splits_replace = splits[-1].replace("\n", "")
                    labels.append(splits_replace)
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    data_name="",
    data_type="",
    is_pmi=False,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_map)
    features = []
    # word_dict = {'B': {}, 'I': {}, 'O': {}}
    with open('/home/minbyul/github/biobert-pytorch/datasets/NER/%s/%s-class_distribution.json' % (data_name, data_type), 'r') as fp:
        word_class_distribution = json.load(fp)
    
    if is_pmi:
        pmi_data = _get_pmi_data(word_class_distribution)
        word_class_distribution = pmi_data

    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        word_class = []
        for word_idx, (word, label) in enumerate(zip(example.words, example.labels)):
            word_tokens = tokenizer.tokenize(word)
            
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                word_class.extend([word] * len(word_tokens))

            # for label_key,label_value in word_dict.items():
            #     if word not in word_dict[label_key]:
            #         word_dict[label_key][word] = 0
            # word_dict[label][word] += 1
            
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            word_class = word_class[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        word_class += [sep_token]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            word_class += [sep_token]
            
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            word_class += [cls_token]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            word_class = [cls_token] + word_class

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if is_pmi:
            word_class_tensor = _get_word_class_distribution(word_class, tokens, word_class_distribution, label_ids, max_seq_length, pad_on_left, num_labels, is_pmi=True)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if 'train' in data_type:
            data_type_ids = [1] * max_seq_length
        else:
            data_type_ids = [0] * max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("data_type_ids: %s", " ".join([str(x) for x in data_type_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
        
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, \
                label_ids=label_ids, bias_tensor=word_class_tensor, data_type=data_type_ids,
            )
        )
    # with open('/home/minbyul/github/biobert-pytorch/datasets/NER/%s/%s-class_distribution.json' % (data_name, data_type), 'w') as out_:
    #     json.dump(word_dict, out_, indent=2)

    return features

def _get_pmi_data(word_class_distribution):
    pmi_data = {'B': {}, 'I':{}, 'O':{}}
    b_num, i_num, o_num = 0, 0, 0
    word_freq = 0
    # Pointwise Mutual Information (PMI)
    # add-100 smoothing
    for class_key in word_class_distribution.keys():
        for key, val in word_class_distribution[class_key].items():
            word_class_distribution[class_key][key] += 100

    for class_key in word_class_distribution.keys():
        if class_key == 'B':
            for key, val in word_class_distribution[class_key].items():
                b_num += word_class_distribution['B'][key]
                i_num += word_class_distribution['I'][key]
                o_num += word_class_distribution['O'][key]
    
    word_freq = b_num+i_num+o_num
    for class_key in word_class_distribution.keys():
        for key, val in word_class_distribution[class_key].items():
            cur_word_freq = word_class_distribution['B'][key] + word_class_distribution['I'][key] + word_class_distribution['O'][key]
            numerator = word_class_distribution[class_key][key] / cur_word_freq

            if class_key == 'B':
                denominator = b_num / word_freq * cur_word_freq / word_freq
            elif class_key == 'I':
                denominator = i_num / word_freq * cur_word_freq / word_freq
            else:
                denominator = o_num / word_freq * cur_word_freq / word_freq

            pmi_data[class_key][key] = np.log(numerator / denominator)

    return pmi_data
        

def _get_word_class_distribution(word_class, tokens, word_class_distribution, label_ids, max_seq_length, pad_on_left, num_labels, is_pmi=False):
    assert len(word_class) == len(tokens)
        
    class_bias = []
    padding_length = max_seq_length - len(word_class)

    class_bias.append([0] * num_labels) # [CLS]
    for word_idx, word in enumerate(word_class[1:-1]):
        # if need smoothing then add smooth parameter
        if is_pmi:
            class_bias.append([word_class_distribution['B'][word], word_class_distribution['I'][word], word_class_distribution['O'][word]])
        else:
            all_sum = word_class_distribution['B'][word] + word_class_distribution['I'][word] + word_class_distribution['O'][word]
            class_bias.append([word_class_distribution['B'][word]/all_sum, word_class_distribution['I'][word]/all_sum, word_class_distribution['O'][word]/all_sum])
            
    class_bias.append([0] * num_labels) # [SEP]
    if pad_on_left:
        class_bias = ([0] * num_labels)
    else:
        class_bias += ([([0] * num_labels) for i in range(padding_length)])
    
    class_bias = torch.FloatTensor(class_bias).cuda()
    return class_bias

def get_bio_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            # labels = [i+'-bio' if i != 'O' else 'O' for i in labels]
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        return ["O", "B", "I"]

def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

