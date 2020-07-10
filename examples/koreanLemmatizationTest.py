
import pyTextMiner as ptm

korean_lemmatizer=ptm.lemmatizer.KoreanLemmatizer()

test = [
('모', '았다'),
('하', '다'),
('서툰', ''),
('와서', ''),
('내려논', ''),
]

for l, r in test:
    print('({}, {}) -> {}'.format(l, r, korean_lemmatizer(l + r)))
    # print(_lemma_candidate(l, r), end='\n\n')
