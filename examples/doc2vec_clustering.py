from collections import namedtuple

from sklearn.cluster import KMeans

from py_doc2vec.doc2vecModel import Doc2VecTrainer, Doc2VecSimilarity
import logging
import pyTextMiner as ptm
import csv
import sys
from py_document_clustering.documentclustering import DocumentClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model_file = './tmp/1595123417030_pv_dma_dim=100_window=5_epochs=20/doc2vec.model'
doc2vec = Doc2VecSimilarity()
doc2vec.load_model(model_file)
model = doc2vec.get_model()
# name either k-means, agglo, spectral_cocluster
name = 'spectral_cocluster'
clustering = DocumentClustering(k=3)
# n_components means the number of words to be used as features
clustering.make_matrix(n_components=-1, doc2vec_matrix=model.docvecs.vectors_docs)
clustering.cluster(name)

clustering.visualize()
