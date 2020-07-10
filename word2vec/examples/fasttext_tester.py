from word2vec.word_embeddings import FastText

fasttext = FastText()
binary=True
model_file = 'fasttext.bin'
fasttext.load_model(model_file)
mode = 'jamo_split'
print(fasttext.most_similar(mode, positives=['이재명', '경제'], negatives=['정치인'], topn=10))
#print(fasttext.most_similar(mode, positives=['이재명'], negatives=[], topn=10))

print('-----------------------------------')

print(fasttext.similar_by_word(mode, '이재명'))