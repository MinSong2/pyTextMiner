import pyTextMiner as ptm

pipeline = None
corpus = ptm.CorpusFromFieldDelimitedFile('./data/donald.txt',2)
mecab_path = 'C:\\mecab\\mecab-ko-dic'
mode = 'korean_lemmatizer'
if mode is not 'korean_lemmatizer':
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            #ptm.tokenizer.Komoran(),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1,2,concat=' '),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))
else :
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            #ptm.tokenizer.Komoran(),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            # ptm.ngram.NGramTokenizer(1, 2, concat=' '))
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))


documents = ['오늘은 비가와서 그런지 매우 우울하다',
             '시험이 끝나야 놀지 스트레스 받아ㅠㅠ',
             '행복한 하루의 끝이라 아름답고 좋네!',
             '더운날에는 아이스 커피가 최고지~~!']

#result = pipeline.processCorpus(corpus)
result = pipeline.processCorpus(documents)
print(result)