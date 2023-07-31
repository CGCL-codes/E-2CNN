import json
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
from relation.data import add_marker_tokens


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.entity_type = json.load(open(config.entity_path, 'r'))
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_name)
        add_marker_tokens(self.tokenizer, self.entity_type)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_relations)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text, marker_encoded_text, marker_entity):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)

        # 在加了entity marker的句子中，marker的embedding与原始句子中对应的 entity_markers_tokens[batch, seq, dim]
        entity_markers_tokens = torch.matmul(marker_entity, marker_encoded_text)
        # sub_head和sub_tail在加了marker的句子中，对应的marker的embedding
        marker_sub_head = torch.matmul(sub_head_mapping, entity_markers_tokens)
        marker_sub_tail = torch.matmul(sub_tail_mapping, entity_markers_tokens)
        # sub = (sub_head + marker_sub_head + sub_tail + marker_sub_tail) / 4
        sub = (sub_head + sub_tail) / 2
        encoded_text = encoded_text + sub
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails
    
    def get_encoded_type_text(self, encoded_text, marker_encoded_text, marker_entity):
        marker_emb = torch.matmul(marker_entity, marker_encoded_text)
        encoded_type_text = encoded_text + marker_emb
        return encoded_type_text


    def forward(self, token_ids, mask, sub_head, sub_tail, marker_token_ids, marker_masks, marker_entity):
        encoded_text = self.get_encoded_text(token_ids, mask)
        marker_encoded_text = self.get_encoded_text(marker_token_ids, marker_masks)
        encoded_type_text =  self.get_encoded_type_text(encoded_text, marker_encoded_text, marker_entity)     
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_type_text)
        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_type_text, marker_encoded_text, marker_entity)

        return {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
        }