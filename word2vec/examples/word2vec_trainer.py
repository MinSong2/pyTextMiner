
from word2vec.word_embeddings import Word2Vec

word2vec = Word2Vec()
mode = 'simple'
mecab_path = 'C:\\mecab\\mecab-ko-dic'
stopword_file = '../stopwords/stopwordsKor.txt'
files = []
files.append('../data/donald.txt')
is_directory=False
doc_index=2
max=-1
word2vec.preprocessing(mode,mecab_path,stopword_file,files,is_directory,doc_index,max)

min_count=1
window=5
size=50
negative=5
word2vec.train(min_count, window, size, negative)

model_file = 'word2vec.bin'
binary=True;
word2vec.save_model(model_file, binary)


