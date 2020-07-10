
import pyTextMiner as ptm

#corpus = ptm.CorpusFromFieldDelimitedFile('./data/donald.txt', 2)
corpus=ptm.CorpusFromFile('./data/134963_norm.txt')
# import nltk
# nltk.download()
# 단어 단위로 분리했으니 이제 stopwords를 제거하는게 가능합니다. ptm.helper.StopwordFilter를 사용하여 불필요한 단어들을 지워보도록 하겠습니다.
# 그리고 파이프라인 뒤에 ptm.stemmer.Porter()를 추가하여 어근 추출을 해보겠습니다.
# 한번 코드를 고쳐서 ptm.stemmer.Lancaster()도 사용해보세요. Lancaster stemmer가 Porter stemmer와 어떻게 다른지 비교하면 재미있을 겁니다.
pipeline = ptm.Pipeline(ptm.splitter.NLTK(), ptm.tokenizer.Komoran(),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))
result = pipeline.processCorpus(corpus)
print(result)
print()

documents=[]
for doc in result:
    document=''
    for sent in doc:
        document = " ".join(sent)
    documents.append(document)

#2016-10-20.txt
corpus1=ptm.CorpusFromFile('./data/2016-10-20.txt')
noun_extractor=ptm.noun_extractor.NounExtractionKorean(corpus1)
sent='두바이월드센터시카고옵션거래소'
result=noun_extractor.__call__(sent)
print(result)