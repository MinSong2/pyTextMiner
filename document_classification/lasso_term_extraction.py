import json
from collections import namedtuple

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pyTextMiner as ptm

KeywordScore = namedtuple('KeywordScore', 'word frequency coefficient')

class LassoTermExtractor:

    def __init__(self, min_tf=20, min_df=10, costs=None, verbose=True):
        name = 'lasso'
        self.min_tf = min_tf
        self.min_df = min_df
        self.costs = [500, 200, 100, 50, 10, 5, 1, 0.1] if costs == None else costs
        self.costs = sorted(self.costs)
        self.verbose = verbose
        self.index2word = None
        self.word2index = None
        self._tfs = {}

    def train(self, x, index2word=None, word2index=None):
        self.num_doc, self.num_term = x.shape
        self.index2word = index2word
        self.word2index = word2index

        rows, cols = x.nonzero()
        b = csr_matrix(([1] * len(rows), (rows, cols)))
        _df = dict(enumerate(b.sum(axis=0).tolist()[0]))
        _df = {word: df for word, df in _df.items() if df >= self.min_df}

        self._tfs = dict(enumerate(x.sum(axis=0).tolist()[0]))
        self._tfs = {word: freq for word, freq in self._tfs.items() if (freq >= self.min_df) and (word in _df)}

        rows_ = []
        cols_ = []
        data_ = []
        for r, c, d in zip(rows, cols, x.data):
            if not (c in self._tfs):
                continue
            rows_.append(r)
            cols_.append(c)
            data_.append(d)
        self.x = csr_matrix((data_, (rows_, cols_)))
        self._is_empty = [1 if float(d[0]) == 0 else 0 for d in self.x.sum(axis=1)]

    def extract_from_word(self, word, min_num_of_keywords=5):
        word = self._encoding(word)
        pos_idx = self.get_document_index(word)
        return self.extract_from_docs(pos_idx, min_num_of_keywords, except_words={word})

    def _encoding(self, word):
        if type(word) == str:
            if not self.word2index:
                raise ValueError('You should set index2word first')
            word = self.word2index.get(word, -1)
        return word

    def get_document_index(self, word):
        word = self._encoding(word)
        if not (0 <= word < self.num_term):
            return []
        return self.x[:, word].nonzero()[0].tolist()

    def extract_from_docs(self, docs_idx, min_num_of_keywords=5, except_words=None):
        pos_idx = set(docs_idx)
        print(str(len(pos_idx)))
        #for d in pos_idx:
            #print(str(self._is_empty[d]) + " :: " + str(d))
        y = [1 if (d in pos_idx and self._is_empty[d] == 0) else -1 for d in range(self.num_doc)]
        if (1 in set(y)) == False:
            if self.verbose:
                print('There is no corresponding documents')
            return []

        if except_words:
            data_ = []
            rows_ = []
            cols_ = []
            rows, cols = self.x.nonzero()
            for r, w, d in zip(rows, cols, self.x.data):
                if w in except_words:
                    continue
                data_.append(d)
                rows_.append(r)
                cols_.append(w)
            x_ = csr_matrix((data_, (rows_, cols_)))
        else:
            x_ = self.x

        for c in self.costs:
            logistic = LogisticRegression(penalty='l1', C=c, solver='liblinear')
            logistic.fit(x_, y)
            coefficients = logistic.coef_.reshape(-1)
            keywords = sorted(enumerate(coefficients), key=lambda x: x[1], reverse=True)
            keywords = [(word, self._tfs.get(word, 0), coef) for word, coef in keywords if coef > 0]
            logistic = None
            if self.verbose:
                print('%d keywords extracted from %.3f cost' % (len(keywords), c))
            if len(keywords) >= min_num_of_keywords:
                break
        if self.index2word:
            keywords = [KeywordScore(self.index2word[word] if 0 <= word < self.num_term else 'Unk%d' % word, tf, coef)
                        for word, tf, coef in keywords]
        else:
            keywords = [KeywordScore(word, tf, coef) for word, tf, coef in keywords]
        return keywords


if __name__ == '__main__':

    input_file = './data/3_class_naver_news.csv'
    # 1. text processing and representation
    corpus = ptm.CorpusFromFieldDelimitedFileForClassification(input_file,
                                                               delimiter=',',
                                                               doc_index=4,
                                                               class_index=1,
                                                               title_index=3)
    tups = corpus.pair_map
    class_list = []
    for id in tups:
        # print(tups[id])
        class_list.append(tups[id])

    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )
    result = pipeline.processCorpus(corpus)
    print('==  ==')

    with open("./model/id_to_category.json") as handle:
        id_to_category = json.loads(handle.read())

    # 0 - economy 1 - IT 2 - politics
    category = []
    documents = []
    idx = 0
    for doc in result:
        document = ''
        for sent in doc:
            document += " ".join(sent)
        documents.append(document)
        label = class_list[idx]

        if label == id_to_category[str(2)]:
            category.append(idx)
        idx += 1

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(documents)
    print(x.shape)

    word2index = vectorizer.vocabulary_
    index2word = sorted(
        vectorizer.vocabulary_,
        key=lambda x: vectorizer.vocabulary_[x]
    )

    lasso = LassoTermExtractor(costs=[500, 200, 100, 50, 10, 5, 1, 0.1],
                               min_tf=10,
                               min_df=5)
    lasso.train(x,index2word,word2index)
    #keywords = lasso.extract_from_word(137,min_num_of_keywords=30)

    keywords = lasso.extract_from_docs(category,min_num_of_keywords=30)
    print(keywords[:20])
