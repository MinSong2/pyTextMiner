import pyLDAvis.gensim
import pickle
import gensim

class ldaInference:
    def __init__(self, dictionary_model='dictionary.gensim', corpus_model='corpus.pkl', lda_model='model5.gensim'):
        self.dictionary = gensim.corpora.Dictionary.load(dictionary_model)
        self.corpus = pickle.load(open(corpus_model, 'rb'))
        self.lda = gensim.models.ldamodel.LdaModel.load(lda_model)

    def infer(self, document):
        test_doc = [self.dictionary.doc2bow(document.split(" "))]
        inferred_matrix = self.lda.inference(test_doc)

        return inferred_matrix

if __name__ == '__main__':
    a_document = '한국 시장경제가 위기입니다.'
    inferred_topics = ldaInference().infer(a_document)

    print(str(inferred_topics))
