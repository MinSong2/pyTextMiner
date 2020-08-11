from pytorch_pretrained_bert import BertTokenizer

import py_ner.bert_ner_prediction as prediction
import torch
import py_ner.lstm_cnn_crf_utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
json_file = './models/tag_info_kr.json' #tag_info_kr.json or tag_info_en.json
tag2idx = utils.load_from_json(json_file)
MAX_LEN = 160

#text = 'this is a good John Smith as my friend'
text = '이승만 대통령은 대한민국 박명환 대통령입니다.'

#bert-base-multilingual-cased, bert-base-cased
tokenizer_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

predictor = prediction.BERTNERPredictor()
model_name = "bert_ner_kr.model" #bert_ner_kr.model or bert_ner_en.model
predictor.load_model(model_name)
predictions = predictor.predict_each(device, text, tokenizer, MAX_LEN, tag2idx)
#pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
print(str(predictions))