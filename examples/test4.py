import tomotopy as tp
print(tp.isa) # 'avx2'나 'avx', 'sse2', 'none'를 출력합니다.

import pyTextMiner as ptm
import io
import nltk

mecab_path = 'C:\\mecab\\mecab-ko-dic'
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab(mecab_path),
                        #ptm.tokenizer.Komoran(),
                        ptm.lemmatizer.SejongPOSLemmatizer(),
                        ptm.helper.SelectWordOnly(),
                        #ptm.ngram.NGramTokenizer(1, 2, concat=' '))
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))

documents = ['오늘은 비가와서 그런지 매우 우울한 날이다',
             '시험이 끝나야 놀지 스트레스 받아ㅠㅠ',
             '행복한 하루의 끝이라 좋네!']

corpus = ptm.CorpusFromFieldDelimitedFile('./data/donald.txt',2)
#result = pipeline.processCorpus(corpus)

result = pipeline.processCorpus(documents)
print(result)


from soylemma import Lemmatizer
lemmatizer = Lemmatizer(dictionary_name='default')
re = lemmatizer.lemmatize('밝은')
print('result ' + str(re))

test_list = ['http://www.google.com', "why", "ftpfjdjkwjkjw", "no no!"]
PROTOCOLS = ('http', 'https', 'ftp', 'git')
for s in test_list:
    if s.startswith(tuple(p for p in PROTOCOLS)):
        print("true " + s)
    else:
        print("false " + s)

