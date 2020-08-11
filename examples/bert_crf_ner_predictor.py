from py_ner.bert_crf_ner_prediction import BertCRFNERPredictor

model_dir = '../py_ner/experiments/base_model_with_crf_val'
predictor = BertCRFNERPredictor(model_dir)

model_name = 'bert_lstm_crf'
tokenizer_path = "../py_ner/ptr_lm_model/tokenizer_78b3253a26.model"
checkpoint_file = '../py_ner/experiments/base_model_with_crf_val/best-epoch-9-step-750-acc-0.980.bin'
predictor.load_model(model_name=model_name, tokenizer_path=tokenizer_path, checkpoint_file=checkpoint_file)

text = '김대중 대통령은 노벨평화상을 받으러 스웨덴으로 출국해서 5박6일 동안 스웨덴에 머물며 대한민국의 위상을 높였다.'
ne_text = predictor.predict(text)
print(ne_text)