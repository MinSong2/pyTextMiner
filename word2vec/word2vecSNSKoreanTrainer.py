import multiprocessing
from time import time
import gensim
from gensim.models.word2vec import LineSentence

import pyTextMiner as ptm
from gensim.models import Word2Vec

from gensim.test.utils import datapath

documents = []

path = '/usr/local/lib/mecab/dic/mecab-ko-dic'
cores = multiprocessing.cpu_count() # Count the number of cores in a computer

print('Start reading the dataset....')
mode = 'simple' #mode is either filtered or unfiltered or simple
if mode == 'unfiltered':
    pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                            ptm.tokenizer.MeCab(path),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))

    corpus1 = ptm.CorpusFromFieldDelimitedFile('/Data/ko_sns_comments/naver_comment_2016_only.txt',1)
    corpus2 = ptm.CorpusFromFieldDelimitedFile('/Data/ko_sns_comments/naver_comment_2015_only.txt',1)

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
    #documents = LineSentence(datapath('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'))
    count = 0
    for line in open('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'):
        toks = line.split()
        if len(toks) > 10:
            documents.append(toks)
            count += 1

        if count % 10000 == 0:
            print('processing... ' + str(count))

print('Document size for the total dataset: ' + str(len(documents)))

model = gensim.models.Word2Vec(min_count=5,
                               window=5,
                               size=300,
                               sample=6e-5,
                               alpha=0.03,
                               min_alpha=0.0007,
                               negative=20,
                               workers=cores-1)

modelFile = './korean_sns_comments_w2v.bin'

t = time()
model.build_vocab(documents, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

model.train(documents, total_examples=model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

model.wv.save_word2vec_format(modelFile, binary=True)