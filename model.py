# coding: utf-8
# @File: model.py
# @Author: HE D.H.
# @Email: victor-he@qq.com
# @Time: 2020/10/10 17:12:56
# @Description:

import torch
import torch.nn as nn
from transformers import  BertModel

class BertClassifier(nn.Module):
    def __init__(self, bert_config):
        super().__init__()
        self.bert = BertModel(config=bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        self.softmax = nn.functional.softmax

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = bert_output[1]
        logits = self.classifier(pooled)
        return self.softmax(logits,1)
