from transformers import BertModel, BertForSequenceClassification
from torch import nn, optim
import torch
import os
from kobert_transformers import get_kobert_model

class PYBERTClassifier(nn.Module):
    '''
     Customized BERT Sequence Model
    '''
    def __init__(self, n_classes, model_name):
        #PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        super(PYBERTClassifier, self).__init__()
        if 'etri' in model_name or 'mecab' in model_name:
            self.bert = BertModel.from_pretrained(os.path.abspath('pytorch_model.bin'),
                                  output_hidden_states = False)
        else:
            self.bert = BertModel.from_pretrained(model_name)

        #print(self.bert.config.hidden_size)

        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #print(pooled_output.shape)

        output = self.drop(pooled_output)
        return self.out(output)

    def name(self):
        return 'PYBERTClassifier'

class PYBERTClassifierGenAtten(nn.Module):
    def __init__(self,
                 n_classes,
                 model_name,
                 dr_rate=None,
                 params=None):

        '''
        bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None
        '''

        super(PYBERTClassifierGenAtten, self).__init__()
        if 'etri' in model_name or 'mecab' in model_name:
            self.bert = BertModel.from_pretrained(os.path.abspath('pytorch_model.bin'),
                                                  output_hidden_states=False)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.dr_rate = dr_rate
        self.attention_mask=None

        if self.dr_rate != None:
            print('dropout ' + str(self.dr_rate))
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, targets):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(targets):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def get_attention_mask(self, atten_mask):
        self.attention_mask = atten_mask

    def forward(self, token_ids, targets, segment_ids, attention_mask):
        if attention_mask is None:
            self.attention_mask = self.gen_attention_mask(token_ids, targets)
        else:
            self.attention_mask = attention_mask

        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long(),
                              attention_mask=self.attention_mask.float().to(token_ids.device))

        if self.dr_rate:
            output = self.dropout(pooler)

        return self.out(output)

    def name(self):
        return 'PYBERTClassifierGenAtten'

class PYBertForSequenceClassification:
    '''
        Use pytorch's BERTForSeqeunceClassification
        Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.
        labels (torch.LongTensor of shape (batch_size,), optional, defaults to None)
        – Labels for computing the sequence classification/regression loss. Indices should be in [0, ..., config.num_labels - 1]. If config.num_labels == 1 a regression loss is computed (Mean-Square loss),
        If config.num_labels > 1 a classification loss is computed (Cross-Entropy).
    '''
    def __init__(self, n_classes, model_name):
        self.model = BertForSequenceClassification.from_pretrained(
                                    model_name,  # Use the 12-layer BERT model, with an uncased vocab.
                                    num_labels=n_classes,  # The number of output labels--2 for binary classification.
                                    # You can increase this for multi-class tasks.
                                    output_attentions=False,  # Whether the model returns attentions weights.
                                    output_hidden_states=False,  # Whether the model returns all hidden-states.
                                )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def __call__(self, *args, **kwargs):
        return self.model

    def name(self):
        return 'PYBertForSequenceClassification'