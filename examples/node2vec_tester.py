import pyTextMiner as ptm
from py_node2vec.node2vecModel import Node2VecModel

embedding_filename='./node2vec.emb'
n2vec = Node2VecModel()
n2vec.load_model(embedding_filename)
results= n2vec.most_similars('정치')
print(results)

pair_similarity = n2vec.compute_similarity('문재인', '정치')
for pair in pair_similarity:
    print(str(pair[0]) + " -- " + str(pair[1]))