import torch
import numpy as np
from torch import nn, optim
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import torch.nn.functional as F
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PYBERTTrainer:
    def __init__(self):
        print('training model with BERT')

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        self.training_stats = []

        self.df_stats = []

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, scheduler, algorithm='transformers'):
        model = model.train()
        correct_predictions = 0.0
        # Reset the total loss for this epoch.
        total_train_loss = 0.0

        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets = d["targets"].to(device)

            if algorithm == 'transformers':
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     labels=targets)

                pred = torch.argmax(F.softmax(logits), dim=1)
                correct = pred.eq(targets)

                label_ids = targets.to('cpu').numpy()

                correct_predictions += correct.sum().item()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            else:
                #print('starting the process based on BertModel API, which is a harder way')
                #using BertModel API directly
                if model.name() == 'PYBERTClassifier':
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                elif model.name() == 'PYBERTClassifierGenAtten':
                    outputs = model(input_ids, targets, token_type_ids,attention_mask)

                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)

                total_train_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_avg = float(correct_predictions) / float(len(data_loader))*10
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(data_loader)

        return avg_train_avg, avg_train_loss

    def flat_accuracy(self, preds, labels):
        '''
         # Function to calculate the accuracy of our predictions vs labels
        :param preds:
        :param labels:
        :return:
        '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def eval_model(self, model, data_loader, loss_fn, device, algorithm='transformers'):
        model = model.eval()

        correct_predictions = 0
        total_eval_accuracy = 0.0
        total_eval_loss = 0.0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                token_type_ids = d["token_type_ids"].to(device)
                targets = d["targets"].to(device)

                if algorithm == 'transformers':
                    (loss, logits) = model(input_ids,
                                           token_type_ids=None,
                                           attention_mask=attention_mask,
                                           labels=targets)

                    # Accumulate the validation loss.
                    total_eval_loss += loss.item()
                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = targets.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += self.flat_accuracy(logits, label_ids)

                else:
                    if model.name() == 'PYBERTClassifier':
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    elif model.name() == 'PYBERTClassifierGenAtten':
                        outputs = model(
                            input_ids,
                            targets,
                            token_type_ids,
                            attention_mask)

                    _, preds = torch.max(outputs, dim=1)
                    loss = loss_fn(outputs, targets)

                    total_eval_accuracy += torch.sum(preds == targets)
                    # Accumulate the validation loss.
                    total_eval_loss += loss.item()

        # Report the final accuracy for this validation run.
        avg_val_accuracy = float(total_eval_accuracy) / float(len(data_loader))*10

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(data_loader)

        return avg_val_accuracy, avg_val_loss

    def train(self, model, device, train_data_loader, val_data_loader, df_val,
              df_train, tokenizer, num_epochs=10, algorithm='transformers',
              torch_model_name='best_model_states.bin'):

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        total_steps = len(train_data_loader) * num_epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_accuracy = 0

        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        for epoch in range(num_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            train_acc, train_loss = self.train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                algorithm=algorithm
            )

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(train_loss))
            print("  Average accuracy: {0:.2f}".format(train_acc))

            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")
            t0 = time.time()

            val_acc, val_loss = self.eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                algorithm=algorithm
            )

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            self.training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': train_loss,
                    'Valid. Loss': val_loss,
                    'Valid. Accur.': val_acc,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

            print("")
            print("Training complete!")

            print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))

            if val_acc > best_accuracy:
                #torch.save(model.state_dict(), saved_training_model)
                self._save(model,tokenizer, algorithm, torch_model_name)
                best_accuracy = val_acc


    def _save(self, model, tokenizer, algorithm, torch_model_name='best_model_states.bin'):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

        output_dir = './model_save/'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        if algorithm == 'transformers':
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        torch.save(model, os.path.join(output_dir, torch_model_name))
        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    def summanry_training_stats(self):
        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=self.training_stats)

        # Use the 'epoch' as the row index.
        self.df_stats = df_stats.set_index('epoch')

        # A hack to force the column headers to wrap.
        # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

        # Display the table.
        print(self.df_stats)

    def visualize_performance(self):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(self.df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(self.df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()