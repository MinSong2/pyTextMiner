import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pyTextMiner as ptm

def vectorizeCaseOne():
    documents = [
                 'This is the first document.',
                 'This document is the second document.',
                 'And this is the third one.',
                 'Is this the first document?',
                ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(X.toarray())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(X.toarray())

def vectorizeCaseTwo():
    corpus = ptm.CorpusFromFieldDelimitedFile('./data/donald.txt',2)

    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt')
                            )
    result = pipeline.processCorpus(corpus)
    print('== 형태소 분석 + 명사만 추출 + 단어만 보여주기 + 빈도 분석 ==')
    print(result)
    print()

    print('==  ==')

    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            document += " ".join(sent)
        documents.append(document)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(X.shape)

    print(X.toarray())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    print(X.toarray())


#vectorizeCaseOne()

vectorizeCaseTwo()

