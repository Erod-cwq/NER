import conllu
from conllu import parse_incr
import json
from transformers import BertTokenizer
import os
from src.utils.dataset_utils import NERDataset
from torch.utils.data import RandomSampler, DataLoader


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class CRFFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None,):
        super(CRFFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        # labels
        self.labels = labels


def get_ent2ids():
    entities = ['prov', 'city', 'district', 'devzone', 'town', 'community', 'village_group', 'road',
                'roadno', 'poi', 'subpoi', 'houseno', 'cellno', 'floorno', 'roomno', 'detail', 'assist',
                'distance', 'intersection']
    ent2id = {
        'O': 1,
    }
    i = 1
    for entity in entities:
        for prefix in ['B', 'I', 'E', 'S']:
            i += 1
            ent2id[prefix + '-' + entity] = i

    with open('ent2ids.json', 'w') as f:
        json.dump(ent2id, f)


def get_examples():
    sentence = []
    current_tag_classes = []
    sentences = []
    tags = []
    with open('ent2ids.json', 'r') as f:
        ent2ids = json.load(f)
    with open('data/train.conll', 'r', encoding="utf-8") as train_dataset:
        for line in train_dataset:
            line = line.strip()
            if line:
                word, tag_class = line.split(' ')
                sentence.append(word)
                if tag_class:
                    current_tag_classes.append(ent2ids[tag_class])
            else:
                sentences.append(sentence)
                tags.append(current_tag_classes)
                assert len(sentence) == len(current_tag_classes)
                sentence = []
                current_tag_classes = []
    assert len(sentences) == len(tags)
    return sentences, tags


def convert_crf_examples(sentence, label_ids, tokenizer):
    max_seq_len = 512
    assert len(sentence) == len(label_ids)
    encode_dict = tokenizer.encode_plus(text=sentence,
                                        max_length=512,
                                        padding='max_length',
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    if len(label_ids) < max_seq_len:
        pad_length = max_seq_len - len(label_ids)
        label_ids = label_ids + [0] * pad_length
    assert len(label_ids) == max_seq_len, f'{len(label_ids)}'

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    feature = CRFFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
    )

    return feature


def get_training_set():
    sentences, tags = get_examples()
    features = []
    tokenizer = BertTokenizer('../bert/usr/vocab.txt')
    for i in range(len(sentences)):
        feature = convert_crf_examples(sentences[i], tags[i], tokenizer)
        features.append(feature)
    train_dataset = NERDataset("crf", features, 'train')

    return train_dataset


