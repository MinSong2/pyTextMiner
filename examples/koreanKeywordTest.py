import pyTextMiner as ptm

min_count = 5   # 단어의 최소 출현 빈도수 (그래프 생성 시)
max_length = 20 # 단어의 최대 길이
beta = 0.95
max_iter = 20
verbose = True
num_words=30
keyword_extractor=ptm.keyword.KeywordExtractionKorean(min_count,max_length,beta,max_iter,verbose,num_words)

sents = ['최순실 씨가 외국인투자촉진법 개정안 통과와 예산안 반영까지 꼼꼼이 챙긴 건데, 이른바 외촉법, 어떤 법이길래 최 씨가 열심히 챙긴 걸까요. 자신의 이해관계와 맞아 떨어지는 부분이 없었는지 취재기자와 한걸음 더 들여다보겠습니다. 이서준 기자, 우선 외국인투자촉진법 개정안, 어떤 내용입니까?',
        '한마디로 대기업이 외국 투자를 받아 계열사를 설립할 때 규제를 완화시켜 주는 법안입니다. 대기업 지주사의 손자 회사가 이른바 증손회사를 만들 때 지분 100%를 출자해야 합니다. 대기업의 문어발식 계열사 확장을 막기 위한 조치인데요. 외촉법 개정안은 손자회사가 외국 투자를 받아서 증손회사를 만들 땐 예외적으로 50% 지분만 투자해도 되게끔 해주는 내용입니다.',
        '그만큼 쉽게 완화해주는 거잖아요. 그때 기억을 더듬어보면 야당의 반발이 매우 심했습니다. 그 이유가 뭐였죠? ',
        '대기업 특혜 법안이라는 취지였는데요. (당연히 그랬겠죠.) 당시 박영선 의원의 국회 발언을 들어보시겠습니다. [박영선 의원/더불어민주당 (2013년 12월 31일) : 경제의 근간을 흔드는 법을 무원칙적으로 이렇게 특정 재벌 회사에게 특혜를 주기 위해 간청하는 민원법을 우리가 새해부터 왜 통과시켜야 합니까.]',
        '최순실 씨 사건을 쫓아가다 보면 본의 아니게 이번 정부의 과거로 올라가면서 복기하는 듯한 느낌이 드는데 이것도 바로 그중 하나입니다. 생생하게 기억합니다. 이 때 장면들은. 특정 재벌 회사를 위한 특혜라고 말하는데, 어떤 기업을 말하는 건가요?',
        'SK와 GS 입니다. 개정안이 통과되는 걸 전제로 두 회사는 외국 투자를 받아 증손회사 설립을 진행중이었기 때문인데요. 당시 개정안이 통과되지 않으면 두 기업이 수조원의 손실이 생길 수 있는 것으로 알려져 있었습니다. 허창수 GS 회장과 김창근 SK회장은 2013년 8월 박 대통령과 청와대에서 대기업 회장단 오찬자리에서 외촉법 통과를 요청한 바도 있습니다. ',
        '물론 두 기업과 최순실 씨와 연결고리가 나온 건 아니지만, 정 전 비서관 녹취파일 속 최 씨는 외촉법에 상당히 집착을 하는 걸로 보이긴 합니다.',
        '네 그렇습니다. 통화 내용을 다시 짚어보면요. 최 씨는 외촉법 관련 예산이 12월 2일, 반드시 되어야 한다, 작년 예산으로 돼서는 안 된다고 얘기하고 있는데요. 다시 말해서 외촉법 관련 예산안이 내년에 반영되어야 한다고 압박을 하고 있는 겁니다. 그러면서 "국민을 볼모로 잡고 있다"며 "국회와 정치권에 책임을 묻겠다"고 으름장까지 놓고 있는데요. 매우 집착하는 모습인데요. 이에 대해서 정 전 비서관이 "예산이 그렇게 빨리 통과된 적 없습니다"고 말하자 말을 끊으면서 매우 흥분한 듯, "그렇더라도, 그렇더라도" 하면서 "야당이 공약 지키라고 하면서 협조는 안 한다", "대통령으로 할 수 있는 일이 없다", "불공정 사태와 난맥상이 나온다"며 굉장한 압박까지 하고 있습니다.',
        '이 얘기들만 들여다봐도 마치 본인이 대통령처럼 얘기하고 있습니다. 내용들 보면 그렇지 않습니까? 혹시 최 씨가 이 외촉법 통과로 이득을 본 경우도 있습니까. ',
        '최 씨가 입김을 넣어 차은택 씨가 주도를 한 걸로 알려진 K컬처밸리 사업이 그렇다는 얘기가 나오고 있습니다. 외촉법을 편법으로 활용해 1% 금리를 적용받았다는 지적이 나오고 있습니다. 본격 사업이 추진되기 전 최순실 국정개입 사건이 터지기는 했지만, 이외에도 다른 혜택을 받았는지는 조사가 필요해 보입니다. ',
        '그런데 녹취파일을 보면 "남자1"이 등장합니다. 이 사람은 누구입니까?',
        '정 전 비서관을 "정 과장님"으로 부르며 반말을 하는 남자인데요. 최순실 씨처럼 정 전 비서관을 하대하고 있습니다. 또 청와대 내부 정보를 알고 있는 듯하고 또 인사에까지 개입하려고 하고 있습니다. 그렇기 때문에 정윤회 씨로 추정은 됩니다만 확인은 되지 않습니다.'
]

keyword=keyword_extractor(sents)
for word, r in sorted(keyword.items(), key=lambda x: x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))

corpus = ptm.CorpusFromFieldDelimitedFile('./data/donald.txt', 2)

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


keyword_extractor1=ptm.keyword.KeywordExtractionKorean(min_count,max_length,beta,max_iter,verbose,num_words)
keyword1=keyword_extractor1(documents)
for word, r in sorted(keyword1.items(), key=lambda x: x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))