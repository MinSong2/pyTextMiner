import pyTextMiner as ptm

#tokenize by subwords
scores = {'파스': 0.3, '파스타': 0.7, '좋아요': 0.2, '좋아':0.5}
tokenizer = ptm.tokenizer.MaxScoreTokenizerKorean(scores=scores)
tokens = tokenizer.inst.tokenize('파스타가좋아요')
print(str(tokens))

#띄어쓰기가 잘 되어 있는 한국어 문서의 경우에는 MaxScoreTokenizer를 이용할 필요가 없다.
# 한국어는 L+[R] 구조이기 때문이다
# 이 때에는 한 어절의 왼쪽에서부터 글자 점수가 가장 높은 부분을 기준으로 토크나이징을 한다
scores = {'데이':0.5, '데이터':0.5, '데이터마이닝':0.5, '공부':0.5, '공부중':0.45}
tokenizer = ptm.tokenizer.LTokenizerKorean(scores=scores)
print('\nflatten=True\nsent = 데이터마이닝을 공부한다')
print(tokenizer.inst.tokenize('데이터마이닝을 공부한다'))

print('\nflatten=False\nsent = 데이터마이닝을 공부한다')
print(tokenizer.inst.tokenize('데이터마이닝을 공부한다', flatten=False))

print('\nflatten=False\nsent = 데이터분석을 위해서 데이터마이닝을 공부한다')
print(tokenizer.inst.tokenize('데이터분석을 위해서 데이터마이닝을 공부한다', flatten=False))

print('\nflatten=True\nsent = 데이터분석을 위해서 데이터마이닝을 공부한다')
print(tokenizer.inst.tokenize('데이터분석을 위해서 데이터마이닝을 공부한다'))

#Tolerance는 한 어절에서 subword 들의 점수의 차이가 그 어절의 점수 최대값과 tolerance 이하로 난다면, 길이가 가장 긴 어절을 선택한다.
# CohesionProbability에서는 합성명사들은 각각의 요소들보다 낮기 때문에 tolerance를 이용할 수 있다.
#
print('tolerance=0.0\nsent = 데이터마이닝을 공부중이다')
print(tokenizer.inst.tokenize('데이터마이닝을 공부중이다'))

print('\ntolerance=0.1\nsent = 데이터마이닝을 공부중이다')
print(tokenizer.inst.tokenize('데이터마이닝을 공부중이다', tolerance=0.1))

#RegexTokenizer는 regular extression을 이용하여 언어가 달라지는 순간들을 띄어쓴다.
# 영어의 경우에는 움라우트가 들어가는 경우들이 있어서 알파벳 뿐 아니라 라틴까지 포함하였다.
tokenizer = ptm.tokenizer.RegexTokenizerKorean()

sents = [
    '이렇게연속된문장은잘리지않습니다만',
    '숫자123이영어abc에섞여있으면ㅋㅋ잘리겠죠',
    '띄어쓰기가 포함되어있으면 이정보는10점!꼭띄워야죠'
]

for sent in sents:
    print('   %s\n->%s\n' % (sent, tokenizer.inst.tokenize(sent)))
