import gensim

modelFile = 'korean_sns_comments_w2v.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=True, unicode_errors='ignore')

print(model.most_similar(positive=['이재명'], topn=10))
print('-----------------------------------')

print(model.similar_by_word('이재명'))



