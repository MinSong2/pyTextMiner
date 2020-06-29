import matplotlib
from gensim import corpora
import pickle
import gensim

class LDAManager:
    def __init__(self, numTopics, numWords):
        name = 'LDA'
        self.numTopics = numTopics
        self.numWords = numWords

    def createDictionary(self, text_data):
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]

        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

        return corpus, dictionary

    def run(self, text_data):
        corpus, dictionary = self.createDictionary(text_data)
        tf_idf = gensim.models.TfidfModel(corpus)
        corpus_tfidf = tf_idf[corpus]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   num_topics=self.numTopics,
                                                   id2word=dictionary,
                                                   passes=15,
                                                   iterations=1500,
                                                   alpha='symmetric',
                                                   per_word_topics=True
                                                   )

        # Get topic weights and dominant topics ------------
        from sklearn.manifold import TSNE
        from bokeh.plotting import figure, output_file, show
        from bokeh.models import Label
        import pandas as pd
        import numpy as np
        import matplotlib.colors as mcolors

        # Get topic weights
        topic_weights = []
        for i, row_list in enumerate(ldamodel[corpus]):
            #print("XXXX " + str(i) + " : " + str(row_list[0]))
            topic_weights.append([w for i, w in row_list[0]])

        # Array of topic weights
        arr = pd.DataFrame(topic_weights).fillna(0).values

        # Keep the well separated points (optional)
        arr = arr[np.amax(arr, axis=1) > 0.35]

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)
        print(str(topic_num) + " :: " + str(arr))

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        # Plot the Topic Clusters using Bokeh
        n_topics = 4
        mycolors = np.array([color for name, color in matplotlib.colors.cnames.items()])
        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                      plot_width=900, plot_height=700)
        plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
        show(plot)

        for topic_id in range(ldamodel.num_topics):
            topk = ldamodel.show_topic(topic_id, 10)
            topk_words = [w for w, _ in topk]
            print('{}: {}'.format(topic_id, ' '.join(topk_words)))

        ldamodel.save('model5.gensim')
        topics = ldamodel.print_topics(self.numWords)
        with io.open("topic_results.txt", 'w', encoding='utf8') as f:
            for topic in topics:
                #print(topic)
                f.write(str(topic) + "\n")

        doc_count = 0
        with io.open("doc_topic_results.txt", 'w', encoding='utf8') as f:
            for _doc in corpus:
                document_topics = ldamodel.get_document_topics(_doc)
                doc_dist = ''
                for aTuple in document_topics:
                    doc_dist += str(aTuple[0]) + " : " + str(aTuple[1]) + " "
                #print(str(text_data[doc_count]) + " " + doc_dist + "\n")
                f.write(str(text_data[doc_count]) + " " + doc_dist  + "\n")
                doc_count += 1

if __name__ == '__main__':
    import pyTextMiner as ptm
    import io
    import nltk

    #corpus = ptm.CorpusFromFile('../data/donald.txt')
    corpus = ptm.CorpusFromFieldDelimitedFile('../data/donald.txt', 2)
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'),
                            ptm.ngram.NGramTokenizer(1,2))

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

    numTopics = 5
    numWords = 15
    LDAManager(numTopics, numWords).run(text_data)

