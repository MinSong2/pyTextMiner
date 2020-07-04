import multiprocessing
from time import time

import gensim
from gensim.models import FastText

import pyTextMiner as ptm

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

print('Start reading the dataset....')
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


fasttext_model = FastText(documents, size=300, window=5, min_count=3,
                     workers=10, sg=1, min_n=2, max_n=6)

fasttext_model.save('./korean_wiki_ft.bin')

t = time()
fasttext_model.build_vocab(documents, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

fasttext_model.train(documents, total_examples=fasttext_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

