import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch.nn.functional as F

from py_bert.bert_dataset import PYBERTDataset


class bert_predictor:

    def __init(self):
        print('BERT Predictor')
        self.prediction_dataloader = None
        self.model = None

    def load_data(self, df, tokenizer, max_len):
        # Create sentence and label lists
        sentences = df.content.values
        labels = df.sentiment.values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Set the batch size.
        batch_size = 32

        # Create the DataLoader.
        #prediction_data = TensorDataset(input_ids, attention_masks, labels)
        #prediction_sampler = SequentialSampler(prediction_data)

        ds = PYBERTDataset(
            contents=df.content.to_numpy(),
            targets=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len)

        prediction_sampler = SequentialSampler(ds)

        self.prediction_dataloader = DataLoader(ds,
                                                sampler=prediction_sampler,
                                                batch_size=batch_size)

        # Prediction on test set
        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        print('Positive samples: %d of %d (%.2f%%)' % (
        df.sentiment.sum(), len(df.sentiment), (df.sentiment.sum() / len(df.sentiment) * 100.0)))

    def load_model(self, saved_training_model):
        self.model = torch.load(saved_training_model)

        # Put model in evaluation mode
        self.model.eval()

    def predict(self, device, algorithm='transformers'):
        contents = []
        predictions = []
        prediction_probs = []
        true_labels = []

        # Predict
        for batch in self.prediction_dataloader:

            # batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            texts = batch["document_text"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            #print(str(len(input_ids)) + ' == ' + str(len(targets)))

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                if algorithm == 'transformers':
                    outputs = self.model(input_ids,
                                         token_type_ids=None,
                                         attention_mask=attention_mask)
                else:
                    if self.model.name() == 'PYBERTClassifier':
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    elif self.model.name() == 'PYBERTClassifierGenAtten':
                        outputs = self.model(input_ids, targets, token_type_ids, attention_mask)
                    #outputs = self.model(
                    #    input_ids=input_ids,
                    #    attention_mask=attention_mask
                    #)

            if algorithm == 'transformers':
                logits = outputs[0]
                _, preds = torch.max(logits, dim=1)
                # Move logits and labels to CPU
                preds = preds.detach().cpu().numpy()
                # The predictions for this batch are a 2-column ndarray (one column for "0"
                # and one column for "1"). Pick the label with the highest value and turn this
                # in to a list of 0s and 1s.
                #probs = F.softmax(logits, dim=1)
                probs = logits
            else:
                _, preds = torch.max(outputs, dim=1)
                preds = preds.detach().cpu().numpy()
                probs = outputs

            targets = targets.to('cpu').numpy()

            # Store predictions and true labels along with text and prediction probability
            # need to know the difference between python list's extend and append
            contents.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            true_labels.extend(targets)

        print(' DONE.')

        return contents, predictions, prediction_probs, true_labels

    def predict_each(self, device, text, tokenizer, MAX_LEN, algorithm='transformers'):

        encoded_text = tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        token_type_ids = encoded_text["token_type_ids"].to(device)

        # Predict
        preds = 0
        probs = None

        # Forward pass, calculate logit predictions
        if algorithm == 'transformers':
            outputs = self.model(input_ids,
                                 token_type_ids=None,
                                 attention_mask=attention_mask)
        else:
            if self.model.name() == 'PYBERTClassifier':
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif self.model.name() == 'PYBERTClassifierGenAtten':

                outputs = self.model(input_ids, None, token_type_ids, attention_mask)

            #outputs = self.model(
            #    input_ids=input_ids,
            #    attention_mask=attention_mask
            #)

        if algorithm == 'transformers':
            logits = outputs[0]
            _, preds = torch.max(logits, dim=1)
            # Move logits and labels to CPU
            preds = preds.detach().cpu().numpy()
            # The predictions for this batch are a 2-column ndarray (one column for "0"
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            # probs = F.softmax(logits, dim=1)
            probs = logits
        else:
            _, preds = torch.max(outputs, dim=1)
            preds = preds.detach().cpu().numpy()
            probs = outputs

        print(str(preds) + " :: " + str(probs))
        return preds[0]

    def evaluate(self, labels, predictions):
        matthews_set = []

        # Evaluate each test batch using Matthew's correlation coefficient
        print('Calculating Matthews Corr. Coef. for each batch...')
        # For each input batch...
        for i in range(len(labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0"
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            #pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
            pred_labels_i = predictions[i]
            # Calculate and store the coef for this batch.
            matthews = matthews_corrcoef(labels[i], pred_labels_i)
            matthews_set.append(matthews)

        # Create a barplot showing the MCC score for each batch of test samples.
        ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

        plt.title('MCC Score per Batch')
        plt.ylabel('MCC Score (-1 to +1)')
        plt.xlabel('Batch #')

        plt.show()

        # Combine the results across all batches.
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(labels, axis=0)

        # Calculate the MCC
        mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

        print('Total MCC: %.3f' % mcc)

