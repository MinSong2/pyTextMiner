from gensim import corpora, models, similarities
import pickle
import gensim

class tfidf:
    def __init__(self):
        name = 'tfidf'

    def createDictionary(self, text_data):
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]

        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        return corpus, dictionary

    def run(self, text_data):
        _corpus, dictionary = self.createDictionary(text_data)
        tf_idf = models.TfidfModel(_corpus)  # step 1 -- initialize a model
        corpus_tfidf = tf_idf[_corpus]
        for doc in corpus_tfidf:
            print(doc)

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)  # initialize an LSI transformation
        result = lsi.print_topics(5,20)
        for a_topic in result:
            print("LSI results " + str(a_topic))

        corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
        #for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
            #print(doc)

if __name__ == '__main__':
    import pyTextMiner as ptm
    import io
    import nltk

    corpus = ptm.CorpusFromFile('../donald.txt')
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(), ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='../stopwordsKor.txt'),
                            ptm.ngram.NGramTokenizer(3))

    result = pipeline.processCorpus(corpus)

    id = 0
    text_data = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        text_data.append(new_doc)
        id += 1

    tfidf().run(text_data)