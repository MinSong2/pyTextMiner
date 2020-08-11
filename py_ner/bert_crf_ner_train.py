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
from py_ner.data_utils.utils import CheckpointManager, SummaryManager
from py_ner.model.net import KobertCRF
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

class BertCrfTrainer:
    def __init__(self, data_dir='data', model_dir='experiments/base_model_with_crf_val'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.logger = logging.getLogger(__name__)
        self.model_config = None
        self.tr_ds = None
        self.val_ds = None
        self.data_dir = data_dir
        self.model_dir = model_dir

    def set_seed(self, seed=100):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def data_loading(self):

        # data_config = Config(json_path=data_dir / 'config.json')
        self.model_config = Config(json_path=self.model_dir + '/config.json')

        # Vocab & Tokenizer
        tok_path = get_tokenizer()  # ./tokenizer_78b3253a26.model
        ptr_tokenizer = SentencepieceTokenizer(tok_path)

        _, vocab_of_gluonnlp = get_pytorch_kobert_model()
        token_to_idx = vocab_of_gluonnlp.token_to_idx

        self.model_config.vocab_size = len(token_to_idx)
        self.vocab = Vocabulary(token_to_idx=token_to_idx)

        print("len(token_to_idx): ", len(token_to_idx))
        with open(self.model_dir + "/token2idx_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(token_to_idx, f, ensure_ascii=False, indent=4)

        # save vocab & tokenizer
        with open(self.model_dir + "/vocab.pkl", 'wb') as f:
            pickle.dump(self.vocab, f)

        # load vocab & tokenizer
        with open(self.model_dir + "/vocab.pkl", 'rb') as f:
            self.vocab = pickle.load(f)

        tokenizer = Tokenizer(vocab=self.vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=self.model_config.maxlen)
        ner_formatter = NamedEntityRecognitionFormatter(vocab=self.vocab, tokenizer=tokenizer, maxlen=self.model_config.maxlen,
                                                        model_dir=self.model_dir)

        # Train & Val Datasets
        cwd = Path.cwd()
        data_in = self.data_dir
        train_data_dir = data_in + "/말뭉치 - 형태소_개체명"
        tr_ds = NamedEntityRecognitionDataset(train_data_dir=train_data_dir, model_dir=self.model_dir)
        tr_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn,
                               transform_target_fn=ner_formatter.transform_target_fn)
        self.tr_ds = tr_ds
        self.tr_dl = DataLoader(tr_ds, batch_size=self.model_config.batch_size, shuffle=True, num_workers=0, drop_last=False)

        val_data_dir = data_in + "/validation_set"
        val_ds = NamedEntityRecognitionDataset(train_data_dir=val_data_dir, model_dir=self.model_dir)
        val_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn,
                                transform_target_fn=ner_formatter.transform_target_fn)
        self.val_ds = val_ds
        self.val_dl = DataLoader(val_ds, batch_size=self.model_config.batch_size, shuffle=True, num_workers=0, drop_last=False)

    def train(self):
        # Model
        model = KobertCRF(config=self.model_config, num_classes=len(self.tr_ds.ner_to_index))
        model.to(self.device)
        model.train()

        # optim
        train_examples_len = len(self.tr_ds)
        val_examples_len = len(self.val_ds)
        print("num of train: {}, num of val: {}".format(train_examples_len, val_examples_len))

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # num_train_optimization_steps = int(train_examples_len / model_config.batch_size / model_config.gradient_accumulation_steps) * model_config.epochs
        t_total = len(self.tr_dl) // self.model_config.gradient_accumulation_steps * self.model_config.epochs

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_config.learning_rate, eps=self.model_config.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.model_config.warmup_steps, t_total=t_total)

        n_gpu = torch.cuda.device_count()
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        # save
        tb_writer = SummaryWriter('{}/runs'.format(self.model_dir))
        checkpoint_manager = CheckpointManager(self.model_dir)
        summary_manager = SummaryManager(self.model_dir)

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(self.tr_ds))
        self.logger.info("  Num Epochs = %d", self.model_config.epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.model_config.batch_size)
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
            epoch_iterator = tqdm(self.tr_dl, desc="Iteration")
            epoch = _epoch
            for step, batch in enumerate(epoch_iterator):
                model.train()
                x_input, token_type_ids, y_real = map(lambda elm: elm.to(self.device), batch)
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
                        sequence_of_tags = torch.tensor(sequence_of_tags).to(self.device)
                        print(sequence_of_tags.shape)
                        _tags = torch.squeeze(sequence_of_tags, dim=0)
                        mb_acc = (_tags == y_real).float()[y_real != self.vocab.PAD_ID].mean()

                    tr_acc = mb_acc.item()
                    tr_loss_avg = tr_loss / global_step
                    tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}

                    # if step % 50 == 0:
                    print('epoch : {}, global_step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, global_step,
                                                                                                 tr_summary['loss'],
                                                                                                 tr_summary['acc']))

                    # training & evaluation log
                    if self.model_config.logging_steps > 0 and global_step % self.model_config.logging_steps == 0:
                        if self.model_config.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            eval_summary, list_of_y_real, list_of_pred_tags = self.evaluate(model, self.val_dl)
                            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                            tb_writer.add_scalars('loss',
                                                  {'train': (tr_loss - logging_loss) / self.model_config.logging_steps,
                                                   'val': eval_summary["eval_loss"]}, global_step)
                            tb_writer.add_scalars('acc', {'train': tr_acc, 'val': eval_summary["eval_acc"]},
                                                  global_step)
                            print("eval acc: {}, loss: {}, global steps: {}".format(eval_summary['eval_acc'],
                                                                                    eval_summary['eval_loss'],
                                                                                    global_step))
                        print("Average loss: {} at global step: {}".format(
                            (tr_loss - logging_loss) / self.model_config.logging_steps, global_step))
                        logging_loss = tr_loss

                    # save model
                    if self.model_config.save_steps > 0 and global_step % self.model_config.save_steps == 0:
                        eval_summary, list_of_y_real, list_of_pred_tags = self.evaluate(model, self.val_dl)

                        # Save model checkpoint
                        output_dir = os.path.join(self.model_config.output_dir, 'epoch-{}'.format(epoch + 1))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        print("Saving model checkpoint to %s", output_dir)
                        state = {'global_step': global_step + 1,
                                 'model_state_dict': model.state_dict(),
                                 'opt_state_dict': optimizer.state_dict()}
                        summary = {'train': tr_summary, 'eval': eval_summary}
                        summary_manager.update(summary)
                        print("summary: ", summary)
                        summary_manager.save('summary.json')

                        # Save
                        is_best = eval_summary["eval_acc"] >= best_dev_acc  # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)
                        if is_best:
                            best_dev_acc = eval_summary["eval_acc"]
                            best_dev_loss = eval_summary["eval_loss"]
                            best_steps = global_step
                            # if args.do_test:
                            # results_test = evaluate(model, test_dl, test=True)
                            # for key, value in results_test.items():
                            #     tb_writer.add_scalar('test_{}'.format(key), value, global_step)
                            # logger.info("test acc: %s, loss: %s, global steps: %s", str(eval_summary['eval_acc']), str(eval_summary['eval_loss']), str(global_step))

                            checkpoint_manager.save_checkpoint(state,
                                                               'best-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1,
                                                                                                             global_step,
                                                                                                             best_dev_acc))
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
                                                                                                          eval_summary[
                                                                                                              "eval_acc"])))
                            print("Saving model checkpoint as model-epoch-{}-step-{}-acc-{:.3f}.bin".format(epoch + 1,
                                                                                                            global_step,
                                                                                                            eval_summary[
                                                                                                                "eval_acc"]))

        tb_writer.close()
        print("global_step = {}, average loss = {}".format(global_step, tr_loss / global_step))

    def evaluate(self, model, prefix="NER"):
        """ evaluate accuracy and return result """
        results = {}
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Eval!
        self.logger.info("***** Running evaluation {} *****".format(prefix))
        eval_loss = 0.0
        nb_eval_steps = 0

        list_of_y_real = []
        list_of_pred_tags = []
        count_correct = 0
        total_count = 0

        for batch in tqdm(self.val_dl, desc="Evaluating"):
            model.train()
            x_input, token_type_ids, y_real = map(lambda elm: elm.to(device), batch)
            with torch.no_grad():
                inputs = {'input_ids': x_input,
                          'token_type_ids': token_type_ids,
                          'tags': y_real}
                log_likelihood, sequence_of_tags = model(**inputs)

                eval_loss += -1 * log_likelihood.float().item()
            nb_eval_steps += 1

            y_real = y_real.to('cpu')
            sequence_of_tags = torch.tensor(sequence_of_tags).to('cpu')
            _tag = torch.squeeze(sequence_of_tags, dim=0)
            count_correct += (_tag == y_real).float()[y_real != 2].sum()  # 0,1,2,3 -> [CLS], [SEP], [PAD], [MASK] index
            total_count += len(y_real[y_real != 2])

            for seq_elm in y_real.tolist():
                list_of_y_real += seq_elm

            for seq_elm in _tag.tolist():
                list_of_pred_tags += seq_elm

            # for seq_elm in sequence_of_tags.tolist():
            # list_of_pred_tags += seq_elm

        eval_loss = eval_loss / nb_eval_steps
        acc = (count_correct / total_count).item()  # tensor -> float
        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        return results, list_of_y_real, list_of_pred_tags


    def save_cr_and_cm(self, list_of_y_real, list_of_pred_tags, cr_save_path="classification_report.csv",
                       cm_save_path="confusion_matrix.png"):
        """ print classification report and confusion matrix """

        # target_names = val_dl.dataset.ner_to_index.keys()
        sorted_ner_to_index = sorted(self.val_dl.dataset.ner_to_index.items(), key=operator.itemgetter(1))
        target_names = []
        for ner_tag, index in sorted_ner_to_index:
            if ner_tag in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', 'O']:
                continue
            else:
                target_names.append(ner_tag)

        label_index_to_print = list(
            range(5, 25))  # ner label indice except '[CLS]', '[SEP]', '[PAD]', '[MASK]' and 'O' tag
        print(classification_report(y_true=list_of_y_real, y_pred=list_of_pred_tags, target_names=target_names,
                                    labels=label_index_to_print, digits=4))
        cr_dict = classification_report(y_true=list_of_y_real, y_pred=list_of_pred_tags, target_names=target_names,
                                        labels=label_index_to_print, digits=4, output_dict=True)
        df = pd.DataFrame(cr_dict).transpose()
        df.to_csv(cr_save_path)
        np.set_printoptions(precision=2)
        self.plot_confusion_matrix(y_true=list_of_y_real, y_pred=list_of_pred_tags, classes=target_names,
                              labels=label_index_to_print, normalize=False,
                              title='Confusion matrix, without normalization')
        plt.savefig(cm_save_path)
        # plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes, labels,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        # Only use the labels that appear in the data

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # --- plot 크기 조절 --- #
        plt.rcParams['savefig.dpi'] = 200
        plt.rcParams['figure.dpi'] = 200
        plt.rcParams['figure.figsize'] = [20, 20]  # plot 크기
        plt.rcParams.update({'font.size': 10})
        # --- plot 크기 조절 --- #

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # --- bar 크기 조절 --- #
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        # --- bar 크기 조절 --- #
        # ax.figure.colorbar(im, ax=ax)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax
