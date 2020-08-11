from pytorch_pretrained_bert import BertTokenizer

import py_ner.bert_ner_train as train
import torch
import py_ner.lstm_cnn_crf_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = train.BERTNERTrainer()

mode = 'txt'
if mode == 'csv':
    data = "../py_ner/data/ner.csv"
else:
    #for Korean text for now
    data = '../py_ner/data/train.txt'

sentences, tag2idx, labels = trainer.data_processing(data)

#bert-base-multilingual-cased, bert-base-cased
tokenizer_name = 'bert-base-multilingual-cased'
tokenizer = trainer.tokenizer(tokenizer_name)
trainer.data_loading(tokenizer,sentences,tag2idx,labels)
classifier_model_name='bert-base-multilingual-cased'
trainer.load_token_classifier(classifier_model_name,tag2idx)

trainer.set_optimizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer.train_epoch(device)

trainer.eval(device, labels)

language = "kr"
trainer.save_model(language)
