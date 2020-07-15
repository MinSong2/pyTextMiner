import platform

import numpy as np
from gensim import models
import matplotlib.font_manager as fm
import tabulate

#for visualization
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sPCA
from sklearn import manifold #MSD, t-SNE

class visualizeW2V:
    def __init__(self):
        self.vf = '../embeddings/word2vec/korean_wiki_w2v.bin'
        self.vecs = models.KeyedVectors.load_word2vec_format(self.vf, binary=True, unicode_errors='ignore')
        self.vecs.init_sims(True)

    def show_closest_line(self, word, n):
        print("words most similar to " + word)

        tops = self.vecs.similar_by_word(word, topn=n, restrict_vocab=None)

        items = [item[0] for item in tops]
        sims = [item[1] for i, item in enumerate(tops)]

        fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)

        font_path = ''
        if platform.system() == 'Windows':
            # Window의 경우 폰트 경로
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        elif platform.system() == 'Darwin':
            # for Mac
            font_path = '/Library/Fonts/AppleGothic.ttf'

        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)

        plt.xticks(range(n), [i + 1 for i in range(n)])
        plt.xlabel('Rank')
        plt.ylabel('Similarity')
        plt.xlim(-1, n)

        ax.plot(sims, color="purple", alpha=0.5)

        for item, x, y in zip(items, range(n), sims):
            ax.annotate(item, xy=(x, y), xytext=(20, -7), textcoords='offset points',
                        ha='right', va='bottom', color='orange', fontsize=14)

        plt.show()

    def show_closest_2d(self, word, n, method):
        tops = self.vecs.similar_by_word(word, topn=n, restrict_vocab=None)
        items = [word] + [x[0] for x in tops]

        wvecs = np.array([self.vecs.word_vec(wd, use_norm=True) for wd in items])

        if method is "PCA":
            spca = sPCA(n_components=2)
            coords = spca.fit_transform(wvecs)
            # print('Explained variation per principal component:', spca.explained_variance_ratio_, "Total:", sum(spca.explained_variance_ratio_))

        elif method is "tSNE":
            tsne = manifold.TSNE(n_components=2)
            coords = tsne.fit_transform(wvecs)
            # print("kl-divergence: %0.8f" % tsne.kl_divergence_)

        elif method == "tSNE-PCA":
            tsne = manifold.TSNE(n_components=2, init='pca')
            coords = tsne.fit_transform(wvecs)
            # print("kl-divergence: %0.8f" % tsne.kl_divergence_)

        elif method is "MDS":
            dists = np.zeros((len(items), len(items)))
            for i, item1 in enumerate(items):
                for j, item2 in enumerate(items):
                    dists[i][j] = dists[j][i] = self.vecs.distance(item1, item2)

            mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0, dissimilarity="precomputed",
                               n_jobs=1)
            coords = mds.fit(dists).embedding_
            # print("Stress is %0.8f" % mds.stress_)

        else:
            raise ValueError("Invalid method: %s" % method)

        font_path = ''
        if platform.system() == 'Windows':
            # Window의 경우 폰트 경로
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        elif platform.system() == 'Darwin':
            # for Mac
            font_path = '/Library/Fonts/AppleGothic.ttf'

        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)

        plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False)

        lim = max([abs(x) for x in coords[:, 0] + coords[:, 1]])
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        plt.scatter(coords[2:, 0], coords[2:, 1])
        plt.scatter(coords[0:1, 0], coords[0:1, 1], color='black')
        plt.scatter(coords[1:2, 0], coords[1:2, 1], color='orange')

        for item, x, y in zip(items[2:], coords[2:, 0], coords[2:, 1]):
            plt.annotate(item, xy=(x, y), xytext=(-2, 2), textcoords='offset points',
                         ha='right', va='bottom', color='purple', fontsize=14)

        x0 = coords[0, 0]
        y0 = coords[0, 1]
        plt.annotate(word, xy=(x0, y0), xytext=(-2, 2), textcoords='offset points',
                     ha='right', va='bottom', color='black', fontsize=16)

        x1 = coords[1, 0]
        y1 = coords[1, 1]
        plt.annotate(items[1], xy=(x1, y1), xytext=(-2, 2), textcoords='offset points',
                     ha='right', va='bottom', color='orange', fontsize=14)

        ax = plt.gca()

        r = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        circle = plt.Circle((x0, y0), r, color='orange', fill=False)
        ax.add_artist(circle)

        plt.show()


    def compare_words_with_color(self, wds):
        wdsr = wds[:]
        wdsr.reverse()

        vs = [self.vecs.get_vector(wd) for wd in wds]
        dim = len(vs[0])

        font_path = ''
        if platform.system() == 'Windows':
            # Window의 경우 폰트 경로
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        elif platform.system() == 'Darwin':
            # for Mac
            font_path = '/Library/Fonts/AppleGothic.ttf'

        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)

        fig = plt.figure(num=None, figsize=(12, 2), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        ax.set_facecolor('gray')

        for i, v in enumerate(vs):
            ax.scatter(range(dim), [i] * dim, c=vs[i], cmap='Spectral', s=16)

        # plt.xticks(range(n), [i+1 for i in range(n)])
        plt.xlabel('Dimension')
        plt.yticks(range(len(wds)), wds)

        plt.show()


    def compare_words_polyline(self, wds, combined=True):
        vs = [self.vecs.get_vector(wd) for wd in wds]
        dim = len(vs[0])
        nseries = len(wds)

        colormap = plt.cm.tab20b
        colors = [colormap(i) for i in np.linspace(0, 1, nseries)]
        font_path = ''
        if platform.system() == 'Windows':
            # Window의 경우 폰트 경로
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        elif platform.system() == 'Darwin':
            # for Mac
            font_path = '/Library/Fonts/AppleGothic.ttf'

        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False)

        if combined:
            fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)

            for i, v in enumerate(vs):
                ax.plot(v, label=wds[i], c=colors[i])

            ax.legend(loc=2)

        else:
            fig, axarr = plt.subplots(nseries + 1, sharex=True, sharey=True, figsize=(12, 2 + nseries), dpi=80,
                                      facecolor='w', edgecolor='k')

            for i, v in enumerate(vs):
                axarr[i + 1].plot(v, label=wds[i], c=colors[i])

            axarr[0].axis('off')

            fig.legend(loc=9)
            fig.tight_layout()

        plt.xlabel('Dimension')
        plt.ylabel('Value')

        plt.show()

if __name__ == '__main__':
    vis = visualizeW2V()

    vis.show_closest_line('사기꾼',10)

    vis.show_closest_2d('사기꾼',10,'tSNE-PCA') #PCA, MDS, tSNE, tSNE-PCA

    words = ['사기꾼','도둑','정치인','연예인']
    words.reverse()
    vis.compare_words_with_color(words)

    words2 = words[:]
    words2.reverse()
    vis.compare_words_polyline(words2, combined=True)
    vis.compare_words_polyline(words2,combined=False)