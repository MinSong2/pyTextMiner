

from word2vec.word_embeddings import GloVe

glove = GloVe()
binary=True
model_file = '../glove-win_devc_x64/vectors.txt'
glove.load_model(model_file)
print(glove.most_similars(positives=['이재명', '경제'], negatives=['정치인'], topn=10))

print('-----------------------------------')

print(glove.most_similar('이재명'))