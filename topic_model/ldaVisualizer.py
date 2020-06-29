import pyLDAvis.gensim
import pickle
import gensim

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
#pyLDAvis.display(lda_display)

pyLDAvis.save_html(lda_display, 'vis.html')

from gensim.test.utils import common_corpus

print(str(common_corpus))