import multiprocessing
from time import time

import gensim
import pyTextMiner as ptm
from gensim.models import Word2Vec

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

print('Start reading the dataset 1....')
path = '/usr/local/lib/mecab/dic/mecab-ko-dic'

pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                        ptm.tokenizer.MeCab(path),
                        ptm.lemmatizer.SejongPOSLemmatizer(),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))

corpus = ptm.CorpusFromFieldDelimitedEmojiFile('/Data/ko_sns_comments/xab',1)
result1 = pipeline.processCorpus(corpus)

print ('Finish processing... ')

i = 0
file = open("naver_comments15_16_filtered.txt", "a+")
for doc in result1:
    if i % 10000 == 0:
        print('processing ' + str(i))
    i += 1
    document = ''
    for sent in doc:
        for word in sent:
            document += word + ' '
    file.write(document.strip() + '\n')

file.close()
print('Document size for the total dataset: ' + str(i))

