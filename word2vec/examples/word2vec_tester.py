
from word2vec.word_embeddings import Word2Vec

word2vec = Word2Vec()
binary=True
model_file = 'word2vec.bin'
word2vec.load_model(model_file, binary)

print(word2vec.most_similar(positives=['이재명', '경제'], negatives=['정치인'], topn=10))
print('-----------------------------------')

print(word2vec.similar_by_word('이재명'))
