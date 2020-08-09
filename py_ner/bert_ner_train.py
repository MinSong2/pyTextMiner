import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from seqeval.metrics import f1_score
import os
from py_ner.ner_data_loader import SentenceGetter, convert_to_df, read_file, convert_to_df_for_ko_ner
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange
import py_ner.lstm_cnn_crf_utils as utils
import pickle

class BERTNERTrainer:
    def __init__(self):
        print('BertNERTrainer')
        self.model = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.optimizer = None

    def data_processing(self, data, mode='txt'):
        #mode = 'kr' # kr or en
        if mode == 'csv':
            dataset = convert_to_df(data)
            json_file = './models/tag_info_en.json'
        elif mode == 'txt':
            print("here!!!!!!")
            #data = './data/train.txt'
            json_file = './models/tag_info_kr.json'
            train_sents = read_file(data)
            dataset, tag_names = convert_to_df_for_ko_ner(train_sents)

            print(dataset.info())
            getter = SentenceGetter(dataset)
            sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
            print(sentences[0])

            labels = [[s[2] for s in sent] for sent in getter.sentences]

            tags_vals = list(set(dataset["tag"].values))
            tag2idx = {t: i for i, t in enumerate(tags_vals)}
            print(str(sentences[0]) + ' --> ' + str(labels[0]))
            print(str(tags_vals) + " :: " + str(tag2idx))

            utils.save_to_json(tag2idx, json_file)

        return sentences, tag2idx, labels

    def tokenizer(self, tokenizer_name):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        return tokenizer

    def data_loading(self, tokenizer, sentences, tag2idx, labels):
        MAX_LEN = 40
        bs =64

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print(tokenized_texts[0])

        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                             maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                             dtype="long", truncating="post")

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                    random_state=2018, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                     random_state=2018, test_size=0.1)

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        self.valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    def load_token_classifier(self, classifier_model_name, tag2idx):
        self.model = BertForTokenClassification.from_pretrained(classifier_model_name, num_labels=len(tag2idx))

    def set_optimizer(self):
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        self.optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train_epoch(self, device):
        epochs = 5
        max_grad_norm = 1.0

        for _ in trange(epochs, desc="Epoch"):
            # TRAIN loop
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = torch.tensor(b_input_ids).to(device).long()
                b_labels = torch.tensor(b_labels).to(device).long()

                # forward pass
                loss = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask, labels=b_labels)
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                # update parameters
                self.optimizer.step()
                self.model.zero_grad()
                # print train loss per epoch
            print("Train loss: {}".format(tr_loss / nb_tr_steps))

    def eval(self, device, tags_vals):
        # VALIDATION on validation set
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in self.valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = torch.tensor(b_input_ids).to(device).long()
            b_labels = torch.tensor(b_labels).to(device).long()

            with torch.no_grad():
                tmp_eval_loss = self.model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

    def save_model(self, language):
        if language == 'kr' or language == 'ko':
            pickle.dump(self.model, open('bert_ner_kr.model', 'wb'))
        else:
            pickle.dump(self.model, open('bert_ner_en.model', 'wb'))



