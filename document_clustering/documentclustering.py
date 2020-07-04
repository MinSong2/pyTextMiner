from collections import defaultdict

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, SpectralCoclustering
import sys
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from time import time
import numpy as np
from numpy.linalg import svd
import operator

class DocumentClustering:
    def __init__(self, k=5):
        self.name = 'k-means'
        self.k = k
        self.X = None
        self.clustering = None
        self.vectorizer = None
        self.dataset_size=0

    def make_matrix(self, documents, n_components=-1):
        self.vectorizer = TfidfVectorizer()
        #self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(documents)
        self.dataset_size=len(documents)

        if (n_components != -1):
            if n_components > len(self.vectorizer.get_feature_names()):
                n_components = len(self.vectorizer.get_feature_names())
            print('n_components ' + str(n_components))
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            print("Performing dimensionality reduction using LSA")
            t0 = time()
            svd = TruncatedSVD(n_components)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            self.X = lsa.fit_transform(self.X)

            print("done in %fs" % (time() - t0))

            explained_variance = svd.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

            print()

    def cluster(self, cluster_name):
        self.name = cluster_name.strip()
        print('cluster_name ' + self.name)
        if self.name == 'k-means':
            print('cluster_name: ' + self.name)
            self.clustering = KMeans(n_clusters=self.k, init='k-means++', max_iter=500, n_init=1)
            print("Clustering sparse data with %s" % self.clustering)
            t0 = time()
            self.clustering.fit(self.X)
            print("done in %0.3fs" % (time() - t0))
            print()
        elif cluster_name == 'agglo':
            self.clustering = AgglomerativeClustering(n_clusters=self.k, affinity='euclidean', memory=None,
                                                      connectivity=None,
                                                      compute_full_tree='auto',
                                                      linkage='ward',
                                                      distance_threshold=None)

            print("Clustering sparse data with %s" % self.clustering)
            t0 = time()

            #to make dense matrix
            self.X = self.X.toarray()
            self.clustering.fit(self.X)
            print("done in %0.3fs" % (time() - t0))
            print()
        elif self.name == 'spectral_cocluster':
            self.clustering = SpectralCoclustering(n_clusters=self.k,svd_method='arpack', random_state=0)
            print("Clustering sparse data with %s" % self.clustering)
            t0 = time()

            self.clustering.fit(self.X)
            print("done in %0.3fs" % (time() - t0))
            print()

    def print_results(self):
        # print the clustering result
        print(self.name)
        if self.name == 'k-means':
            cluster_labels = self.clustering.labels_
            clustering_dict = self.clustering.__dict__
            clusters = {}
            for document_id, cluster_label in enumerate(cluster_labels):
                if cluster_label not in clusters:
                    clusters[cluster_label] = []
                clusters[cluster_label].append(document_id)
                print(str(cluster_label) + " -- " + str(document_id))
            order_centroids = self.clustering.cluster_centers_.argsort()[:, ::-1]
            terms = self.vectorizer.get_feature_names()
            for i in range(self.k):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind], end='')
                print()

            self.X = self.X.toarray()
            #self.X = csr_matrix(self.X).todense()

            pca_t = PCA().fit_transform(self.X)
            #print(self.clustering.labels_)
            plt.scatter(pca_t[:, 0], pca_t[:, 1], c=self.clustering.labels_, cmap='rainbow')
            plt.show()

        elif self.name == 'agglo':
            cluster_labels = self.clustering.labels_
            clustering_dict = self.clustering.__dict__
            clusters = {}

            for document_id, cluster_label in enumerate(cluster_labels):
                if cluster_label not in clusters:
                    clusters[cluster_label] = []
                clusters[cluster_label].append(document_id)
                #print(str(cluster_label) + " -- " + str(document_id))

            results = self.get_cluster_top_keywords(clusters)
            for _cluster in results:
                key_terms = results[_cluster]
                print("Cluster " + str(_cluster) + " : " + str(len(clusters[_cluster])) + " documents")
                print(key_terms)
            print()

            # The output is a one-dimensional array of N documents corresponding to the clusters
            # assigned to our N data points.
            pca_t = PCA().fit_transform(self.X)
            #print(self.clustering.labels_)
            plt.scatter(pca_t[:, 0], pca_t[:, 1], c=self.clustering.labels_, cmap='rainbow')
            plt.show()

        elif self.name == 'spectral_cocluster':
            target_number=10
            bicluster_ncuts = list(self.bicluster_ncut(i) for i in range(self.k))
            best_idx = np.argsort(bicluster_ncuts)[:target_number]

            feature_names = self.vectorizer.get_feature_names()
            print()
            print("Best biclusters:")
            print("----------------")
            for idx, cluster in enumerate(best_idx):
                n_rows, n_cols = self.clustering.get_shape(cluster)
                cluster_docs, cluster_words = self.clustering.get_indices(cluster)
                if not len(cluster_docs) or not len(cluster_words):
                    continue

                # categories
                counter = defaultdict(int)
                for i in cluster_docs:
                    counter[str(i)] += 1
                cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name) for name, c in self.most_common(counter)[:3])

                # words
                out_of_cluster_docs = self.clustering.row_labels_ != cluster
                out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
                word_col = self.X[:, cluster_words]
                word_scores = np.array(word_col[cluster_docs, :].sum(axis=0) -
                                       word_col[out_of_cluster_docs, :].sum(axis=0))
                word_scores = word_scores.ravel()
                important_words = list(feature_names[cluster_words[i]]
                                       for i in word_scores.argsort()[:-11:-1])

                print("bicluster {} : {} documents, {} words".format(idx, n_rows, n_cols))
                print("categories   : {}".format(cat_string))
                print("words        : {}\n".format(', '.join(important_words)))

            pca_t = PCA().fit_transform(self.X.toarray())
            plt.scatter(pca_t[:, 0], pca_t[:, 1], c=self.clustering.row_labels_, cmap='rainbow')
            plt.show()

    def bicluster_ncut(self, i):
        rows, cols = self.clustering.get_indices(i)
        if not (np.any(rows) and np.any(cols)):
            import sys
            return sys.float_info.max

        row_complement = np.nonzero(np.logical_not(self.clustering.rows_[i]))[0]
        col_complement = np.nonzero(np.logical_not(self.clustering.columns_[i]))[0]
        # Note: the following is identical to X[rows[:, np.newaxis],
        # cols].sum() but much faster in scipy <= 0.16
        weight = self.X[rows][:, cols].sum()
        cut = (self.X[row_complement][:, cols].sum() + self.X[rows][:, col_complement].sum())

        return cut / weight

    def most_common(self, d):
        """Items of a defaultdict(int) with the highest values.
        """
        return sorted(d.items(), key=operator.itemgetter(1), reverse=True)

    def get_cluster_top_keywords(self, clusters, keywords_per_cluster=10):
        """Shows the top k words for each cluster
        Keyword Arguments:
            keywords_per_cluster {int} -- The k words to show for each cluster (default: {10})
        Returns:
            dict of lists -- Returns a dict of {cluster_id: ['top', 'k', 'words', 'for', 'cluster']}
        """
        terms = self.vectorizer.get_feature_names()
        out = {}
        docs_for_cluster = {}
        # self.clusters = 10 clusters,containing the index of the document_vectors document in that cluster, ex len(self.clusters[6]) == 508
        for cluster in clusters:
            # To flatten/combine all documents into one
            docs_for_cluster[cluster] = np.array([self.X[i] for i in clusters[cluster]])
            # Cluster vectors to feature words
            out[cluster] = np.array(terms)[np.flip(np.argsort(docs_for_cluster[cluster]), -1)]
            cluster_shape = out[cluster].shape
            out[cluster] = out[cluster].reshape(cluster_shape[0] * cluster_shape[1])[:keywords_per_cluster].tolist()

        return out