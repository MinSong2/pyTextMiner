# yTextMiner의 파이썬 버전, PyTextMiner를 ptm이라는 이름으로 사용하겠다고 선언합니다
# ptm 역시 파이프라인 구조로 텍스트를 처리합니다.
# 만약 pyTextMiner에 빨간줄이 계속 뜬다면 왼쪽의 Project 트리뷰에서 pyTextMiner가 포함된 폴더를 우클릭하여
# 'Mark Directory as'에서 'Sources Root'를 눌러주도록 합시다.
# 이 패키지가 동작하기 위해서는 konlpy와 nltk라는 라이브러리가 필요합니다. konlpy는 저번에 설치했으므로,
# 이번에는 nltk를 설치해봅시다. pip install nltk로 간단하게 설치하시면 됩니다.
import pyTextMiner as ptm
import io

# 다음은 분석에 사용할 corpus를 불러오는 일입니다. sampleEng.txt 파일을 준비해두었으니, 이를 읽어와봅시다.
# ptm의 CorpusFromFile이라는 클래스를 통해 문헌집합을 가져올 수 있습니다. 이 경우 파일 내의 한 라인이 문헌 하나가 됩니다.
#corpus = ptm.CorpusFromFile('donald.txt')
corpus = ptm.CorpusFromDirectory('./tmp', True)

#corpus, pair_map = ptm.CorpusFromFieldDelimitedFileWithYear('./data/donald.txt')

# 이번에는 PyTextMiner로 한국어 처리를 해보도록 하겠습니다. 한국어의 교착어적인 특성 및 복잡한 띄어쓰기 규칙 때문에
# 공백 기준으로 단어를 분리하는 것에는 한계가 있어서 형태소 분석기를 사용합니다.
# ptm.tokenizer.Komoran나 ptm.tokenizer.TwitterKorean을 사용해 형태소 분석이 가능합니다.
# 형태소 분석 이후 품사가 NN으로 시작하는 명사들만 추출하고, 단어만 골라내 출력하도록 해봅시다.

#import nltk
#nltk.download('punkt')

#pipeline = ptm.Pipeline(ptm.splitter.NLTK(), ptm.tokenizer.Komoran(),
#                        ptm.helper.POSFilter('NN*'),
#                        ptm.helper.SelectWordOnly(),
#                        ptm.ngram.NGramTokenizer(3),
#                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt')
#                        )

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.segmentation.SegmentationKorean('./model/korean_segmentation_model.crfsuite'),
                        ptm.ngram.NGramTokenizer(3),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt')
                        )

result = pipeline.processCorpus(corpus)

with io.open("demofile.csv",'w',encoding='utf8') as f:
    for doc in result:
        for sent in doc:
            f.write('\t'.join(sent) + "\n")

print('== 문장 분리 + 형태소 분석 + 명사만 추출 + 단어만 보여주기 + 구 추출 ==')
print(result)
print()
