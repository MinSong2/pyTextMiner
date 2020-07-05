from time import time

import numpy as np
import os
from random import shuffle
import re
import urllib.request
import zipfile
import multiprocessing
from gensim.models import FastText
import pyTextMiner as ptm

path = '/usr/local/lib/mecab/dic/mecab-ko-dic'

documents = []
print('Start reading the dataset....')
mode = 'simple'  # mode is either filtered or unfiltered or simple or jamo_split
if mode == 'unfiltered':
    pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                            ptm.tokenizer.MeCab(path),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))

    corpus1 = ptm.CorpusFromFieldDelimitedFile('/Data/ko_sns_comments/naver_comment_2016_only.txt', 1)
    corpus2 = ptm.CorpusFromFieldDelimitedFile('/Data/ko_sns_comments/naver_comment_2015_only.txt', 1)

    corpus = corpus1.docs + corpus2.docs
    result = pipeline.processCorpus(corpus)
    for doc in result:
        document = []
        for sent in doc:
            for word in sent:
                document.append(word)
        documents.append(document)

elif mode == 'filtered':
    pipeline = ptm.Pipeline(ptm.tokenizer.Word())
    corpus = ptm.CorpusFromFile('/Data/ko_sns_comments/naver_comments15_16_filtered.txt')
    documents = pipeline.processCorpus(corpus)

elif mode == 'simple':
    # documents = LineSentence(datapath('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'))
    count = 0
    #for line in open('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'):
    for line in open('../data/content.txt'):
        toks = line.split()
        if len(toks) > 10:
            documents.append(toks)
            count += 1

        if count % 10000 == 0:
            print('processing... ' + str(count))

elif mode == 'jamo_split':
    # documents = LineSentence(datapath('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'))
    util = ptm.Utility()
    max = 30000000
    count = 0
    #for line in open('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'):
    for line in open('../data/content.txt'):
        if len(line) < 1:
            continue

        sent = util.jamo_sentence(line)
        toks = sent.split()
        if len(toks) > 10:
            documents.append(toks)
            count += 1

        if count % 10000 == 0:
            print('processing... ' + str(count))

        if max < count:
            break

print('Document size for the total dataset: ' + str(len(documents)))

fasttext_model = FastText(documents, size=300, window=5, min_count=3,
                     workers=10, sg=1, min_n=2, max_n=6)

model_name = './korean_sns_comments_ft.bin'
if mode == 'jamo_split':
    model_name = './korean_sns_comments_jamo_ft.bin'
fasttext_model.save(model_name)
print('sent_count ' + str(count))
