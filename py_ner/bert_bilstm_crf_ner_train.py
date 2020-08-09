from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import numpy as np
import logging
import random
import pickle
import json
import os
from pathlib import Path
import operator
import pandas as pd

import torch

from torch.utils.tensorboard import SummaryWriter # from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim

from tqdm import tqdm, trange

from py_ner.bert_crf_ner_train import BertCrfTrainer
from py_ner.data_utils.utils import CheckpointManager, SummaryManager
from py_ner.model.net import KobertCRF, KobertBiGRUCRF, KobertBiLSTMCRF
from py_ner.model.utils import Config

from py_ner.data_utils.ner_dataset import NamedEntityRecognitionDataset, NamedEntityRecognitionFormatter
from py_ner.data_utils.vocab_tokenizer import Vocabulary, Tokenizer
from py_ner.data_utils.pad_sequence import keras_pad_fn
from gluonnlp.data import SentencepieceTokenizer
from py_ner.kobert.pytorch_kobert import get_pytorch_kobert_model
from py_ner.kobert.utils import get_tokenizer
from sklearn.metrics import classification_report
from pytorch_transformers import AdamW, WarmupLinearSchedule

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class BertBiLstmCrfTrainer(BertCrfTrainer):
    def __init__(self, data_dir='data', model_dir='experiments/base_model_with_crf_val'):
        super().__init__(data_dir,model_dir)

    def data_loading(self):
        super().data_loading()

    def train(self):
        # Model
        model = KobertBiLSTMCRF(config=self.model_config, num_classes=len(self.tr_ds.ner_to_index))
        model.train()

        # optim
        train_examples_len = len(self.tr_ds)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # num_train_optimization_steps = int(train_examples_len / model_config.batch_size / model_config.gradient_accumulation_steps) * model_config.epochs
        t_total = len(self.tr_dl) // self.model_config.gradient_accumulation_steps * self.model_config.epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_config.learning_rate, eps=self.model_config.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.model_config.warmup_steps, t_total=t_total)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        n_gpu = torch.cuda.device_count()
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        model.to(device)

        # save
        tb_writer = SummaryWriter('{}/runs'.format(self.model_dir))
        checkpoint_manager = CheckpointManager(self.model_dir)
        summary_manager = SummaryManager(self.model_dir)
        best_val_loss = 1e+10
        best_train_acc = 0

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.tr_ds))
        self.logger.info("  Num Epochs = %d", self.model_config.epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.model_config.batch_size)
        # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        #                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        self.logger.info("  Gradient Accumulation steps = %d", self.model_config.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_dev_acc, best_dev_loss = 0.0, 99999999999.0
        best_steps = 0
        model.zero_grad()
        self.set_seed()  # Added here for reproductibility (even between python 2 and 3)

        # Train
        train_iterator = trange(int(self.model_config.epochs), desc="Epoch")
        for _epoch, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(self.tr_dl, desc="Iteration")  # , disable=args.local_rank not in [-1, 0]
            epoch = _epoch
            for step, batch in enumerate(epoch_iterator):
                model.train()
                x_input, token_type_ids, y_real = map(lambda elm: elm.to(device), batch)
                log_likelihood, sequence_of_tags = model(x_input, token_type_ids, y_real)

                # loss: negative log-likelihood
                loss = -1 * log_likelihood

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.model_config.gradient_accumulation_steps > 1:
                    loss = loss / self.model_config.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_config.max_grad_norm)
                tr_loss += loss.item()

                if (step + 1) % self.model_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    with torch.no_grad():
                        sequence_of_tags = torch.tensor(sequence_of_tags)
                        print("sequence_of_tags: ", sequence_of_tags)
                        print("y_real: ", y_real)
                        print("loss: ", loss)
                        print("(sequence_of_tags == y_real): ", (sequence_of_tags == y_real))
                        _tags = torch.squeeze(sequence_of_tags, dim=0)
                        mb_acc = (_tags == y_real).float()[y_real != self.vocab.PAD_ID].mean()
                        #mb_acc = (sequence_of_tags == y_real).float()[y_real != self.vocab.PAD_ID].mean()

                    tr_acc = mb_acc.item()
                    tr_loss_avg = tr_loss / global_step
                    tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}

                    # if step % 50 == 0:
                    print('epoch : {}, global_step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, global_step,
                                                                                                 tr_summary['loss'],
                                                                                                 tr_summary['acc']))

                    if self.model_config.logging_steps > 0 and global_step % self.model_config.logging_steps == 0:
                        # Log metrics
                        if self.model_config.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            pass
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / self.model_config.logging_steps, global_step)
                        self.logger.info("Average loss: %s at global step: %s",
                                    str((tr_loss - logging_loss) / self.model_config.logging_steps), str(global_step))
                        logging_loss = tr_loss

                    if self.model_config.save_steps > 0 and global_step % self.model_config.save_steps == 0:

                        eval_summary, list_of_y_real, list_of_pred_tags = self.evaluate(model, self.val_dl)

                        # Save model checkpoint
                        output_dir = os.path.join(self.model_config.output_dir, 'epoch-{}'.format(epoch + 1))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        self.logger.info("Saving model checkpoint to %s", output_dir)

                        state = {'global_step': global_step + 1,
                                 'model_state_dict': model.state_dict(),
                                 'opt_state_dict': optimizer.state_dict()}
                        summary = {'train': tr_summary}
                        summary_manager.update(summary)
                        summary_manager.save('summary.json')

                        is_best = tr_acc >= best_train_acc  # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)
                        # Save
                        if is_best:
                            best_train_acc = tr_acc
                            checkpoint_manager.save_checkpoint(state,
                                                               'best-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1,
                                                                                                             global_step,
                                                                                                             tr_acc))

                            print("Saving model checkpoint as best-epoch-{}-step-{}-acc-{:.3f}.bin".format(epoch + 1,
                                                                                                           global_step,
                                                                                                           best_dev_acc))

                            # print classification report and save confusion matrix
                            cr_save_path = self.model_dir + '/best-epoch-{}-step-{}-acc-{:.3f}-cr.csv'.format(epoch + 1,
                                                                                                              global_step,
                                                                                                              best_dev_acc)
                            cm_save_path = self.model_dir + '/best-epoch-{}-step-{}-acc-{:.3f}-cm.png'.format(epoch + 1,
                                                                                                              global_step,
                                                                                                              best_dev_acc)

                            self.save_cr_and_cm(list_of_y_real, list_of_pred_tags, cr_save_path=cr_save_path,
                                           cm_save_path=cm_save_path)
                        else:
                            torch.save(state, os.path.join(output_dir,
                                                           'model-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1,
                                                                                                          global_step,
                                                                                                          tr_acc)))

        tb_writer.close()
        self.logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)



if __name__ == '__main__':
    data = 'data'
    model_dir = 'experiments/base_model_with_crf_val'
    trainer = BertBiLstmCrfTrainer(data_dir=data, model_dir=model_dir)
    trainer.data_loading()
    trainer.train()
    
