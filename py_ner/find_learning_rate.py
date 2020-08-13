import random

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from transformers import AutoTokenizer
import torch
import numpy as np

from py_ner.data_utils.data_utils import *

from py_ner.data_utils.ner_dataset import read_data_from_file, get_labels, NerDataset
from py_ner.model.net import BertForTokenClassificationCustom
from py_ner.model.optimizers import get_optimizer_with_weight_decay

DATA_TR_PATH = './data/JNLPBA/Genia4ERtask1.iob2'
SEED = 42

# MODEL
MODEL_NAME = 'allenai/scibert_scivocab_cased'
MAX_LEN_SEQ = 128

# Optimization parameters
BATCH_SIZE_TR = 32
LEARNING_RATE = 1e-6
CLIPPING = None
OPTIMIZER = Adam

# get data
training_set = read_data_from_file(DATA_TR_PATH)

# Automatically extract labels and their indexes from data.
labels2ind, labels_count = get_labels(training_set)

# Load data
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
training_set = NerDataset(dataset=training_set,
                          tokenizer=tokenizer,
                          labels2ind=labels2ind,
                          max_len_seq=MAX_LEN_SEQ,
                          bert_hugging=False)


dataloader_tr = DataLoader(dataset=training_set,
                           batch_size=BATCH_SIZE_TR,
                           shuffle=True)

# Seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

legend = []
fig = None

for wd in [0, .1, 1e-2, 1e-3, 1e-4]:
    for dp in [.1, 0.2, .3]:
        nerbert = BertForTokenClassificationCustom.from_pretrained(pretrained_model_name_or_path=MODEL_NAME,
                                                                   num_labels=len(labels2ind),
                                                                   hidden_dropout_prob=dp,
                                                                   attention_probs_dropout_prob=dp)

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = get_optimizer_with_weight_decay(model=nerbert,
                                                    optimizer=OPTIMIZER,
                                                    learning_rate=LEARNING_RATE,
                                                    weight_decay=wd)

        lr_finder = LRFinder(nerbert, optimizer, nn.CrossEntropyLoss(), device='cuda')
        lr_finder.range_test(train_loader=dataloader_tr, end_lr=1, num_iter=100)
        fig = lr_finder.plot(ax=fig)
        legend.append(f"wd: {wd}")

fig.figure.legend(legend, loc='best')
fig.figure.tight_layout()
fig.figure.show()
fig.figure.savefig('lr_finder.png')
