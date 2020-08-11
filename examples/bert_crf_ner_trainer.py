import os

from py_ner.bert_crf_ner_train import BertCrfTrainer

data = '../py_ner/data'

# Directory
directory = "exper/base_model_with_crf"
# Parent Directory path
parent_dir = "./"

# Path
path = os.path.join(parent_dir, directory)

# Create the directory
try:
    os.makedirs(path, exist_ok=True)
    print("Directory '%s' created successfully" % directory)
except OSError as error:
    print("Directory '%s' can not be created")

# we need two mandatory files in this new directory: config.json and ner_to_index.json
model_dir = 'exper/base_model_with_crf'
trainer = BertCrfTrainer(data_dir=data, model_dir=model_dir)
trainer.data_loading()
trainer.train()