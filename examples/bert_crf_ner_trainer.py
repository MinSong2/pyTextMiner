import os

from py_ner.bert_bilstm_crf_ner_train import BertBiLstmCrfTrainer
from py_ner.bert_crf_ner_train import BertCrfTrainer
from shutil import copyfile

data = '../py_ner/data'

def make_dir(directory, parent_dir):
    # Path
    path = os.path.join(parent_dir, directory)

    # Create the directory
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory '%s' created successfully" % directory)
    except OSError as error:
        print("Directory '%s' can not be created")

algorithm = 'bert_gru_crf' #bert_crf, bert_lstm_crf, bert_gru_crf
# we need two mandatory files in this new directory: config.json and ner_to_index.json
if algorithm == 'bert_crf':
    model_dir = 'experiments/base_model_with_crf'
    make_dir(model_dir, "./")
    copyfile('../py_ner/config/config.json', model_dir+"/config.json")
    copyfile('../py_ner/config/ner_to_index.json', model_dir + "/ner_to_index.json")

    trainer = BertCrfTrainer(data_dir=data, model_dir=model_dir)
elif algorithm == 'bert_lstm_crf':
    model_dir = 'experiments/base_model_with_lstm_crf'
    make_dir(model_dir, "./")
    copyfile('../py_ner/config/config.json', model_dir + "/config.json")
    copyfile('../py_ner/config/ner_to_index.json', model_dir + "/ner_to_index.json")

    trainer = BertBiLstmCrfTrainer(data_dir=data, model_dir=model_dir)
elif algorithm == 'bert_gru_crf':
    model_dir = 'experiments/base_model_with_gru_crf'
    make_dir(model_dir, "./")
    copyfile('../py_ner/config/config.json', model_dir + "/config.json")
    copyfile('../py_ner/config/ner_to_index.json', model_dir + "/ner_to_index.json")

    trainer = BertBiLstmCrfTrainer(data_dir=data, model_dir=model_dir)

trainer.data_loading()
trainer.train()