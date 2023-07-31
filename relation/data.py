import copy
import json
from random import choice
from fastNLP import TorchLoaderIter, DataSet, Vocabulary, Sampler
from fastNLP.io import JsonLoader
import torch
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_rel = 18


def load_data(train_path, dev_path, test_path, rel_dict_path):
    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}
    loader = JsonLoader({"text": "text", "spo_list": "spo_list", "predict_entity": "predict_entity"})
    data_bundle = loader.load(paths)
    id2rel = json.load(open(rel_dict_path))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))
    return data_bundle, rel_vocab


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

'''
在tokenizer的vocab中加入实体标签
'''
def add_marker_tokens(tokenizer, entity_type):
    new_tokens = []
    for ent_type in entity_type.values():
        new_tokens.append('[%s]'%ent_type)
        new_tokens.append('[/%s]'%ent_type)
    tokenizer.add_tokens(new_tokens)

'''
添加的实体标签不包括subject、object信息
'''
def add_entity_type(json_data, entity_type):
    marker_sent = {}
    entity_set = set()
    marker_sent['text'] = json_data['text']
    marker_sent['spo_list'] = []
    predict_entity = set()
    for entity in json_data['predict_entity']:
        predict_entity.add(json_data['text'][entity[0]:entity[1] + 1])
    for rel in json_data['spo_list']:
        new_subject = "[{}]".format(entity_type[rel['subject_type']]) + rel['subject'] + "[/{}]".format(entity_type[rel['subject_type']])
        new_object = "[{}]".format(entity_type[rel['object_type']]) + rel['object'] + "[/{}]".format(entity_type[rel['object_type']])    
        if rel['subject'] not in entity_set:
            marker_sent['text'] = marker_sent['text'].replace(rel['subject'], new_subject)
            entity_set.add(rel['subject'])
        if rel['object'] not in entity_set:
            marker_sent['text'] = marker_sent['text'].replace(rel['object'], new_object)
            entity_set.add(rel['object'])
    return marker_sent, predict_entity

class MyDataset(DataSet):
    def __init__(self, config, dataset, rel_vocab, is_test):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.rel_vocab = rel_vocab
        self.entity_type = json.load(open(config.entity_path, 'r'))
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_name)
        add_marker_tokens(self.tokenizer, self.entity_type)

    def __getitem__(self, item):
        json_data = self.dataset[item]
        text = json_data['text']
        tokenized = self.tokenizer(text, max_length=self.config.max_len, truncation=True)
        tokens = tokenized['input_ids']
        masks = tokenized['attention_mask']
        text_len = len(tokens)

        token_ids = torch.tensor(tokens, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.bool)
        sub_heads, sub_tails = torch.zeros(text_len), torch.zeros(text_len)
        sub_head, sub_tail = torch.zeros(text_len), torch.zeros(text_len)
        ent_heads, ent_tails = torch.zeros(text_len), torch.zeros(text_len)
        obj_heads = torch.zeros((text_len, self.config.num_relations))
        obj_tails = torch.zeros((text_len, self.config.num_relations))

        if not self.is_test:
            s2ro_map = defaultdict(list)
            for spo in json_data['spo_list']:
                triple = (self.tokenizer(spo['subject'], add_special_tokens=False)['input_ids'],
                          self.rel_vocab.to_index(spo['predicate']),
                          self.tokenizer(spo['object'], add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))

            if s2ro_map:
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

        # 加上了entity marker的数据
        marker_sent, predict_ner = add_entity_type(json_data=json_data, entity_type=self.entity_type)
        marker_text = marker_sent['text']
        marker_tokenized = self.tokenizer(marker_text, max_length=self.config.max_len, truncation=True)
        marker_tokens = marker_tokenized['input_ids']
        marker_masks = marker_tokenized['attention_mask']
        marker_text_len = len(marker_tokens)

        # 保存在加了marker的句子中，实体标签的index与原始句子中对应的实体的index
        marker_entity = torch.zeros((text_len, marker_text_len))
        ent_list = []
        for ele in json_data['predict_entity']:
            pred_ent = json_data['text'][ele[0]:ele[1]+1].replace(' ', '').lower()
            pred_token = self.tokenizer(pred_ent, add_special_tokens=False)['input_ids']
            entity_head_idx = find_head_idx(tokens, pred_token)
            pred_entity_head_idx = find_head_idx(marker_tokens, pred_token)
            marker_entity[entity_head_idx][pred_entity_head_idx - 1] = 1
            marker_entity[entity_head_idx + len(pred_token) - 1][pred_entity_head_idx + len(pred_token)] = 1
            if pred_ent not in ent_list:
                ent_heads[entity_head_idx] = 1
                ent_tails[entity_head_idx + len(pred_token)] = 1
                ent_list.append(pred_ent)
        marker_token_ids = torch.tensor(marker_tokens, dtype=torch.long)
        marker_masks = torch.tensor(marker_masks, dtype=torch.bool)

        triples = []
        for ele in json_data['spo_list']:
            ele['subject'] = ele['subject'].replace(' ', '').lower()
            ele['object'] = ele['object'].replace(' ', '').lower()
            triples.append(ele)

        return token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, text, \
            marker_token_ids, marker_masks, marker_entity

    def __len__(self):
        return len(self.dataset)


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    token_ids, masks, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, texts, marker_token_ids, marker_masks, marker_entity = zip(*batch)
    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)
    batch_sub_heads = pad_sequence(sub_heads, batch_first=True)
    batch_sub_tails = pad_sequence(sub_tails, batch_first=True)
    batch_sub_head = pad_sequence(sub_head, batch_first=True)
    batch_sub_tail = pad_sequence(sub_tail, batch_first=True)
    batch_obj_heads = pad_sequence(obj_heads, batch_first=True)
    batch_obj_tails = pad_sequence(obj_tails, batch_first=True) 
    batch_marker_token_ids = pad_sequence(marker_token_ids, batch_first=True)
    batch_marker_masks = pad_sequence(marker_masks, batch_first=True)
    marker_entity = [pad(marker_ent, pad=[0, batch_marker_masks.size(1) - marker_ent.size(1), 0, 0]) for marker_ent in marker_entity]
    batch_marker_entity = pad_sequence(marker_entity, batch_first=True)

    return {"token_ids": batch_token_ids.to(device),
            "mask": batch_masks.to(device),
            "sub_head": batch_sub_head.to(device),
            "sub_tail": batch_sub_tail.to(device),
            "sub_heads": batch_sub_heads.to(device),
            "text": texts, 
            "marker_token_ids": batch_marker_token_ids.to(device),
            "marker_masks": batch_marker_masks.to(device),
            "marker_entity": batch_marker_entity.to(device),
            }, \
           {"mask": batch_masks.to(device),
            "sub_heads": batch_sub_heads.to(device),
            "sub_tails": batch_sub_tails.to(device),
            "obj_heads": batch_obj_heads.to(device),
            "obj_tails": batch_obj_tails.to(device),
            "triples": triples
            }


class MyRandomSampler(Sampler):
    def __call__(self, data_set):
        return np.random.permutation(len(data_set)).tolist()


def get_data_iterator(config, dataset, rel_vocab, is_test=False, collate_fn=my_collate_fn):
    dataset = MyDataset(config, dataset, rel_vocab, is_test)
    return TorchLoaderIter(dataset=dataset,
                           collate_fn=collate_fn,
                           batch_size=config.batch_size if not is_test else 1,
                           sampler=MyRandomSampler())
