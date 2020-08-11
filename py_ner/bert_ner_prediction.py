
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from torch import nn

import py_ner.lstm_cnn_crf_utils as utils
import pickle

class BERTNERPredictor:
    def __init__(self):
        print('BertNERPredictor')
        self.model = None

    def load_model(self, model_name):
        # open a file, where you stored the pickled data
        file = open(model_name, 'rb')
        # dump information to that file
        self.model = pickle.load(file)
        # close the file
        file.close()

    def getKeyByValue(self, dictOfElements, value):
        key = ''
        listOfItems = dictOfElements.items()
        for item in listOfItems:
            if item[1] == value:
                key = item[0]
        return key

    def align_predictions(self, items, predictions: np.ndarray, label_ids: np.ndarray):
        """Formats the predictions."""
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.getKeyByValue(items, [label_ids[i][j]]))
                    preds_list[i].append(self.getKeyByValue(items, [preds[i][j]]))
        return preds_list, out_label_list

    def predict_each(self, device, text, tokenizer, MAX_LEN, items):

        tokenized_texts = tokenizer.tokenize(text)
        print(tokenized_texts)

        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        pred_masks = [[float(i > 0) for i in ii] for ii in input_ids]
        pred_tags = [[int(i > 0) for i in ii] for ii in input_ids]

        #print(pred_tags)
        pred_ids = torch.tensor(input_ids)
        pred_tags = torch.tensor(pred_tags)
        pred_masks = torch.tensor(pred_masks)

        real_ids = np.argmax(pred_masks, axis=1).tolist()
        print(str(len(real_ids)))

        pred_data = TensorDataset(pred_ids, pred_masks, pred_tags)
        pred_sampler = RandomSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=1)
        i = 0
        predictions = []
        for batch in pred_dataloader:
            if i > 0:
                break
            i += 1
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = torch.tensor(b_input_ids).to(device).long()
            b_labels = torch.tensor(b_labels).to(device).long()

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)

            logits = logits.detach().cpu().numpy()

            pred_flat = np.argmax(logits, axis=2).flatten()

            '''
            for m, a in enumerate(pred_flat):
                if m >= len(real_ids):
                    break
                predictions.append(a)
            '''
            #print(predictions)
            #print(str(len(predictions)))

        #preds_list, out_label_list = self.align_predictions(items, logits, pred_tags)
        #print(preds_list)
        #print(out_label_list)

        return pred_flat
