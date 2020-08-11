from py_ner.bert_crf_ner_prediction import BertCRFNERPredictor

model_dir = '../py_ner/experiments/base_model_with_crf_val'
predictor = BertCRFNERPredictor(model_dir)


tokenizer_path = "./ptr_lm_model/tokenizer_78b3253a26.model"

#model name needs to be changed to the one you trained
model_name = 'best-epoch-9-step-750-acc-0.980.bin'

algorithm = 'bert_lstm_crf'
if algorithm == 'bert_crf':
    checkpoint_file = './experiments/base_model_with_crf/' + model_name

elif algorithm == 'bert_lstm_crf':
    checkpoint_file = './experiments/base_model_with_lstm_crf/' + model_name

elif algorithm == 'bert_gru_crf':
    checkpoint_file = './experiments/base_model_with_gru_crf/' + model_name


predictor.load_model(model_name=model_name, tokenizer_path=tokenizer_path, checkpoint_file=checkpoint_file)

text = '오늘은 비도 오고 학생들이 졸려 보여서 나도 졸리운데 송강호의 괴물 영화나 볼까?'
ne_text = predictor.predict(text)
print(ne_text)