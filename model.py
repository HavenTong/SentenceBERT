import torch
import torch.nn as nn
from transformers import BertModel


class SentenceBERT(nn.Module):
    def __init__(self, config):
        super(SentenceBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.bert_config = self.bert.config
        self.fc = nn.Linear(self.bert_config.hidden_size * 3, config.num_classes)

    def forward(self,
                sen_a_input_ids,
                sen_a_token_type_ids,
                sen_a_attention_mask,
                sen_b_input_ids,
                sen_b_token_type_ids,
                sen_b_attention_mask):
        sen_a_bert_outputs = self.bert(
            input_ids=sen_a_input_ids,
            token_type_ids=sen_a_token_type_ids,
            attention_mask=sen_a_attention_mask
        )
        sen_b_bert_outputs = self.bert(
            input_ids=sen_b_input_ids,
            token_type_ids=sen_b_token_type_ids,
            attention_mask=sen_b_attention_mask
        )
        sen_a_bert_output, sen_b_bert_output = sen_a_bert_outputs[0], sen_b_bert_outputs[0]
        # (batch_size, seq_len, hidden_size)

        sen_a_pooling, sen_b_pooling = sen_a_bert_output.mean(dim=1), sen_b_bert_output.mean(dim=1)
        # (batch_size, hidden_size)

        hidden = torch.cat([sen_a_pooling, sen_b_pooling, torch.abs(sen_a_pooling - sen_b_pooling)], dim=1)

        return self.fc(hidden)


class BertClassifier(nn.Module):
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.bert_config = self.bert.config
        self.fc = nn.Linear(self.bert_config.hidden_size, config.num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        pooler_output = outputs.pooler_output
        return self.fc(pooler_output)

