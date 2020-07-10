import pyTextMiner as ptm

dictionary_path='./dict/user_dic.txt'
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Komoran(userdic=dictionary_path),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        #ptm.tokenizer.MaxScoreTokenizerKorean(),
                        #ptm.tokenizer.Word(),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))
                        #ptm.ngram.NGramTokenizer(2,3),
                        #ptm.counter.WordCounter())

corpus = ptm.CorpusFromEojiFile('./data/filtered_content.txt')
#result = pipeline.processCorpus(corpus)

#print(result)
print()

import numpy as np
print(np.__version__)

s = "회사 동료 분들과 다녀왔는데 분위기도 좋고 음식도 맛있었어요 다만, 강남 토끼정이 강남 쉑쉑버거 골목길로 쭉 올라가야 하는데 다들 쉑쉑버거의 유혹에 넘어갈 뻔 했답니다 강남역 맛집 토끼정의 외부 모습."


pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter())
corpus = [s]
result = pipeline.processCorpus(corpus)
print(result)