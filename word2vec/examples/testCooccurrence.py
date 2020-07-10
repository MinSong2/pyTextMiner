import pyTextMiner as ptm
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib as mpl

if __name__ == '__main__':
    #pipeline = ptm.Pipeline(ptm.splitter.NLTK(), ptm.chunker.KoreanChunker())

    # 다음은 분석에 사용할 corpus를 불러오는 일입니다. sampleEng.txt 파일을 준비해두었으니, 이를 읽어와봅시다.
    # ptm의 CorpusFromFile이라는 클래스를 통해 문헌집합을 가져올 수 있습니다. 이 경우 파일 내의 한 라인이 문헌 하나가 됩니다.
    corpus = ptm.CorpusFromFieldDelimitedFile('./data/donald.txt',2)

    #import nltk
    #nltk.download()
    # 단어 단위로 분리했으니 이제 stopwords를 제거하는게 가능합니다. ptm.helper.StopwordFilter를 사용하여 불필요한 단어들을 지워보도록 하겠습니다.
    # 그리고 파이프라인 뒤에 ptm.stemmer.Porter()를 추가하여 어근 추출을 해보겠습니다.
    # 한번 코드를 고쳐서 ptm.stemmer.Lancaster()도 사용해보세요. Lancaster stemmer가 Porter stemmer와 어떻게 다른지 비교하면 재미있을 겁니다.
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1,2),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt')
                            )
    result = pipeline.processCorpus(corpus)
    print('== 형태소 분석 + 명사만 추출 + 단어만 보여주기 + 빈도 분석 ==')
    print(result)
    print()

    print('==  ==')

    import re
    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            document = " ".join(sent)
            #for English text to remove special chars
            document = re.sub('[^A-Za-z0-9]+', '', document)

        documents.append(document)
    co = ptm.cooccurrence.CooccurrenceWorker()
    co_result, vocab = co.__call__(documents)

    graph_builder = ptm.graphml.GraphMLCreator()

    #mode is either with_threshold or without_threshod
    mode='with_threshold'

    if mode is 'without_threshold':
        print(str(co_result))
        print(str(vocab))
        graph_builder.createGraphML(co_result, vocab, "test1.graphml")

    elif mode is 'with_threshold':
        cv = CountVectorizer()
        cv_fit = cv.fit_transform(documents)
        word_list = cv.get_feature_names();
        count_list = cv_fit.toarray().sum(axis=0)
        word_hist = dict(zip(word_list, count_list))

        print(str(co_result))
        print(str(word_hist))

        graph_builder.createGraphMLWithThreshold(co_result, word_hist, vocab, "test.graphml",threshold=35.0)
        display_limit=50
        graph_builder.summarize_centrality(limit=display_limit)
        title = '동시출현 기반 그래프'
        file_name='test.png'
        graph_builder.plot_graph(title,file=file_name)
