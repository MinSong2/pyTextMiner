from __future__ import print_function, division
import os
import re
import codecs
import unicodedata
from py_ner.lstm_cnn_crf_utils import create_dico, create_mapping, zero_digits
from py_ner.lstm_cnn_crf_utils import iob2, iob_iobes
import py_ner.lstm_cnn_crf_model as model
import pandas as pd

import string
import random
import numpy as np
import pickle

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters + " .,;'-"
    )

def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<PAD>'] = 10000000
    # dico[';'] = 0
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def pad_seq(seq, max_length, PAD_token=0):
    # add pads
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def get_batch(start, batch_size, datas, singletons=[]):
    input_seqs = []
    target_seqs = []
    chars2_seqs = []

    for data in datas[start:start+batch_size]:
        # pair is chosen from pairs randomly
        words = []
        for word in data['words']:
            if word in singletons and np.random.uniform() < 0.5:
                words.append(1)
            else:
                words.append(word)
        input_seqs.append(data['words'])
        target_seqs.append(data['tags'])
        chars2_seqs.append(data['chars'])

    if input_seqs == []:
        return [], [], [], [], [], []
    seq_pairs = sorted(zip(input_seqs, target_seqs, chars2_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, chars2_seqs = zip(*seq_pairs)

    chars2_seqs_lengths = []
    chars2_seqs_padded = []
    for chars2 in chars2_seqs:
        chars2_lengths = [len(c) for c in chars2]
        chars2_padded = [pad_seq(c, max(chars2_lengths)) for c in chars2]
        chars2_seqs_padded.append(chars2_padded)
        chars2_seqs_lengths.append(chars2_lengths)

    input_lengths = [len(s) for s in input_seqs]
    # input_padded is batch * max_length
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    assert target_lengths == input_lengths
    # target_padded is batch * max_length
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # var is max_length * batch_size
    # input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    # target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    #
    # if use_gpu:
    #     input_var = input_var.cuda()
    #     target_var = target_var.cuda()

    return input_padded, input_lengths, target_padded, target_lengths, chars2_seqs_padded, chars2_seqs_lengths


def random_batch(batch_size, train_data, singletons=[]):
    input_seqs = []
    target_seqs = []
    chars2_seqs = []

    for i in range(batch_size):
        # pair is chosen from pairs randomly
        data = random.choice(train_data)
        words = []
        for word in data['words']:
            if word in singletons and np.random.uniform() < 0.5:
                words.append(1)
            else:
                words.append(word)
        input_seqs.append(data['words'])
        target_seqs.append(data['tags'])
        chars2_seqs.append(data['chars'])

    seq_pairs = sorted(zip(input_seqs, target_seqs, chars2_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, chars2_seqs = zip(*seq_pairs)

    chars2_seqs_lengths = []
    chars2_seqs_padded = []
    for chars2 in chars2_seqs:
        chars2_lengths = [len(c) for c in chars2]
        chars2_padded = [pad_seq(c, max(chars2_lengths)) for c in chars2]
        chars2_seqs_padded.append(chars2_padded)
        chars2_seqs_lengths.append(chars2_lengths)

    input_lengths = [len(s) for s in input_seqs]
    # input_padded is batch * max_length
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    assert target_lengths == input_lengths
    # target_padded is batch * max_length
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # var is max_length * batch_size
    # input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    # target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    #
    # if use_gpu:
    #     input_var = input_var.cuda()
    #     target_var = target_var.cuda()

    return input_padded, input_lengths, target_padded, target_lengths, chars2_seqs_padded, chars2_seqs_lengths

def convert_to_df(text_file):
    dframe = pd.read_csv(text_file, encoding="ISO-8859-1", error_bad_lines=False)
    dataset = dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
                           'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
                           'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
                           'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
                           'prev-prev-word', 'prev-shape', 'prev-word', 'shape'], axis=1)

    return dataset

def convert_to_df_for_ko_ner(sentences):
    pd.set_option('display.max_columns', None)
    document_df = pd.DataFrame()

    max = 500;
    for i, sent in enumerate(sentences):
        for ej_idx, token, postag, label in sent:
            document_df = document_df.append(pd.Series([int(i), token, postag, label]), ignore_index=True)
        i += 1

        if i >= max:
            break

    document_df.columns = ['sentence_idx', 'word', 'pos', 'tag']
    document_df = document_df.set_index('sentence_idx')
    tag_names = document_df['tag'].unique()

    return document_df, tag_names

def read_file(file_name):
    sents = []
    with open(file_name,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for idx,l in enumerate(lines) :
            if l[0]==';' and lines[idx+1][0]=='$':
                this_sent = []
            elif l[0]=='$' and lines[idx-1][0]==';':
                continue
            elif l[0]=='\n':
                sents.append(this_sent)
            else :
                this_sent.append(tuple(l.split()))
    return sents

def word2features(sent, i):
    last_p = sent[i][0]
    mor_idx = sent[i][1]
    word = sent[i][3]
    postag = sent[i][4]
    features = [
        'bias',
        'word=' + word,
        'word[:1]=' + word[:1],
        'word[:2]=' + word[:2],
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'last_p=' + last_p,
        'mor_i=' + mor_idx,
        'len='+str(len(word))
    ]
    if i > 0:
        word1 = sent[i-1][3]
        postag1 = sent[i-1][4]
        features.extend([
            '-1:word=' + word1,
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i+1][3]
        postag1 = sent[i+1][4]
        features.extend([
            '+1:word=' + word1,
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def add_features_to_sent(sent):
    new_sent = []
    this_ej = []
    ej_idx = 1
    morph_idx = 1
    for w in sent:
        if int(w[0]) > ej_idx:
            assert(len(this_ej)>0)
            last_particle = '-'
            if this_ej[-1][3][0]=='J':
                last_particle=this_ej[-1][2]+'/'+this_ej[-1][3]
            this_ej = [(last_particle,)+m for m in this_ej]
            new_sent += this_ej
            this_ej = []
            ej_idx += 1
            morph_idx = 1
        this_ej.append((str(morph_idx),)+w)
        morph_idx += 1
    if len(this_ej) > 0:
        last_particle = '-'
        if this_ej[-1][3][0] == 'J':
            last_particle = this_ej[-1][2] + '/' + this_ej[-1][3]
        this_ej = [(last_particle,) + m for m in this_ej]
        new_sent += this_ej
    return new_sent

def sent2features(sent):
    sent = add_features_to_sent(sent)
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for ej_idx, token, postag, label in sent]

def sent2tokens(sent):
    return [token for ej_idx, token, postag, label in sent]


class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["word"].values.tolist(),
                                                           s['pos'].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
