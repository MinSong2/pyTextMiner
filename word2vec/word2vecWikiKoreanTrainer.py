import multiprocessing
from time import time

import gensim
import pyTextMiner as ptm
from gensim.models import Word2Vec

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

print('Start reading the dataset 1....')
path = '/usr/local/lib/mecab/dic/mecab-ko-dic'
pipeline = ptm.Pipeline(ptm.tokenizer.MaxScoreTokenizerKorean(),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))

corpus = ptm.CorpusFromFile('/Data/wiki_dataset/wiki_pos_tokenizer_without_taginfo.txt')
result1 = pipeline.processCorpus(corpus)

print('Document size for wiki: ' + str(len(result1)))

print('Start reading the dataset 2....')
mode = 'filtered' #mode is either filtered or unfiltered
if mode == 'unfiltered':
    pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                            ptm.tokenizer.MeCab(path),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))
    corpus = ptm.CorpusFromFile('/Data/wiki_dataset/namu_wiki.txt')
    result2 = pipeline.processCorpus(corpus)

    documents = result1

    for doc in result2:
        document = []
        for sent in doc:
            document.append(sent.split())
        documents.append(document)

elif mode == 'filtered':
    pipeline = ptm.Pipeline(ptm.tokenizer.Word())
    corpus = ptm.CorpusFromFile('/Data/wiki_dataset/namu_wiki_pos_filtered.txt')
    result2 = pipeline.processCorpus(corpus)

    documents = result1 + result2

print('Document size for the total dataset: ' + str(len(documents)))

model = gensim.models.Word2Vec(min_count=5,
                        window=5,
                        size=300,
                        sample=6e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=20,
                        workers=cores-1)

modelFile = './korean_wiki_w2v.bin'

t = time()
model.build_vocab(documents, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

model.train(documents, total_examples=model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

model.wv.save_word2vec_format(modelFile, binary=True)