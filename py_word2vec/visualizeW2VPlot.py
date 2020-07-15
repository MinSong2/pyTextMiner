from __future__ import unicode_literals

from builtins import zip

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim

class visualizeW2VPlot:
    def __init__(self):
        name = 'visualizeW2VPlot'

    def load(self, modelFile):
        model = gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=True, unicode_errors='ignore')
        return model

    def visualizePCA(self, model):
        pyplot.rc('font', family='New Gulim')

        words = ['이재명', '문재인', '승인', '당', '핵', '평화', '정치인', '대표']

        word_vectors = np.vstack([model[w] for w in words])
        twodim = PCA().fit_transform(word_vectors)[:, :2]
        twodim.shape
        plt.figure(figsize=(5, 5))
        plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
        for word, (x, y) in zip(words, twodim):
            plt.text(x, y, word)
        plt.axis('off');

        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('testPCA.png', dpi=100)


    def visualizeTSNE(self, model, word, vector_size):
        pyplot.rc('font', family='New Gulim')

        arr = np.empty((0, vector_size), dtype='f')
        word_labels = [word]

        # get close words
        close_words = model.similar_by_word(word,topn=20)

        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        # display scatter plot
        plt.scatter(x_coords, y_coords)

        for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
        plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
        plt.show()

if __name__ == '__main__':
    model_file = '../embeddings/word2vec/korean_wiki_w2v.bin'
    model = visualizeW2VPlot().load(model_file)
    mode = 't-sne' # t-sne or pca
    if mode == 'pca':
        visualizeW2VPlot().visualizePCA(model)
    elif mode == 't-sne':
        vector_size = 300
        visualizeW2VPlot().visualizeTSNE(model, '이재명', vector_size)