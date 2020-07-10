
from word2vec.word_embeddings import FastText

fasttext = FastText()
mode = 'jamo_split_filtered'
mecab_path = 'C:\\mecab\\mecab-ko-dic'
stopword_file = '../stopwords/stopwordsKor.txt'
files = []
files.append('../data/donald.txt')
is_directory=False
doc_index=2
max=-1
fasttext.preprocessing(mode,mecab_path,stopword_file,files,is_directory,doc_index,max)

min_count=1
window=5
size=50
negative=5
fasttext.train(min_count, window, size, negative)

model_file = 'fasttext.bin'
binary=True;
fasttext.save_model(model_file)


