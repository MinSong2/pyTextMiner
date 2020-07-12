from sklearn.feature_extraction.text import CountVectorizer

import pyTextMiner as ptm
import re
from py_node2vec.node2vecModel import Node2VecModel

mecab_path = 'C:\\mecab\\mecab-ko-dic'
stopword_file = '../stopwords/stopwordsKor.txt'

pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                        ptm.tokenizer.MeCab(mecab_path),
                        ptm.lemmatizer.SejongPOSLemmatizer(),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file=stopword_file))

corpus = ptm.CorpusFromFieldDelimitedFile('../data/donald.txt',2)
result = pipeline.processCorpus(corpus)
#print(result)
#print()

documents = []
for doc in result:
    document = ''
    for sent in doc:
        n_sent = " ".join(sent)
        #for English text to remove special chars
        document += re.sub('[^A-Za-zㄱ-ㅣ가-힣 ]+', '', n_sent)
    documents.append(document)

co = ptm.cooccurrence.CooccurrenceWorker()
co_result, vocab = co.__call__(documents)

cv = CountVectorizer()
cv_fit = cv.fit_transform(documents)
word_list = cv.get_feature_names();
count_list = cv_fit.toarray().sum(axis=0)
word_hist = dict(zip(word_list, count_list))

threshold = 2.0
dimensions=300
walk_length=30
num_walks=200

n2vec = Node2VecModel()

n2vec.create_graph(co_result, word_hist, threshold)
n2vec.train(dimensions, walk_length, num_walks)

embedding_filename='node2vec.emb'
embedding_model_file='node2vec.model'
n2vec.save_model(embedding_filename,embedding_model_file)