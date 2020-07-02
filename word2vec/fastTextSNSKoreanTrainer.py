import numpy as np
import os
from random import shuffle
import re
import urllib.request
import zipfile
import MeCab

import multiprocessing

from gensim.models import FastText
from soynlp.tokenizer import MaxScoreTokenizer
from konlpy.tag import Mecab

path = '/usr/local/lib/mecab/dic/mecab-ko-dic'
mecab = Mecab(path)

tokenizer = MaxScoreTokenizer()
count = 0
# store as list of lists of words
sentences_ted = []
sentences_strings_ted = []
with open('/Data/ko_sns_comments/naver_comment_2016_only.txt', 'r') as file:
    for line in file:
        toks = line.split('t')
        if len(toks) < 2: continue

        sent_str = line.split('\t')[1]
        tokens = tokenizer.tokenize(sent_str)
        pairs = mecab.pos(sent_str)
        tokens = []
        for pair in pairs:
            tokens.append(pair[0])

        if count == 1:
            print(sent_str + " : " + str(tokens))

        #tokens = sent_str.split()
        sentences_ted.append(tokens)
        count += 1

with open('/Data/ko_sns_comments/naver_comment_2015_only.txt', 'r') as file:
    for line in file:
        toks = line.split('t')
        if len(toks) < 2: continue

        sent_str = line.split('\t')[1]
        tokens = tokenizer.tokenize(sent_str)
        if count == 1:
            print(sent_str + " : " + str(tokens))

        sentences_ted.append(tokens)
        count += 1

model_ted = FastText(sentences_ted, size=300, window=5, min_count=3,
                     workers=10, sg=1, min_n=2, max_n=6)

model_ted.save('./korean_sns_comments_ft.bin')
print('sent_count ' + str(count))
