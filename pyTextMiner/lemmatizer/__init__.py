''' tuple => tuple '''

class BaseTagger:
    IN_TYPE = [tuple]
    OUT_TYPE = [tuple]

class WordNet(BaseTagger):
    def __init__(self):
        import nltk
        nltk.download('wordnet')

        from nltk.stem import WordNetLemmatizer
        self.inst = WordNetLemmatizer()

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        from nltk.corpus import wordnet
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def __call__(self, *args, **kwargs):
        tag = WordNet.get_wordnet_pos(args[0][1])
        return self.inst.lemmatize(args[0][0], tag if tag else 'n'), args[0][1]


class KoreanLemmatizer:
    IN_TYPE = [str]
    OUT_TYPE = [list,str]

    def __init__(self):
        import soynlp
        from soynlp.lemmatizer import Lemmatizer
        from soynlp.lemmatizer import lemma_candidate
        test_stems = {
            '깨닫', '불', '묻', '눋', '겯', '믿', '묻', '뜯', '받',  # ㄷ 불규칙
            '구르', '무르', '마르', '누르', '나르', '모르', '이르',  # 르 불규칙
            '아니꼽', '우습', '더럽', '아름답', '잡', '뽑', '곱', '돕', '새롭', '더럽',  # ㅂ 불규칙
            '낫', '긋', '붓', '뭇', '벗', '솟', '치솟', '씻', '손씻', '뺏',  # ㅅ 불규칙
            '똥푸', '주', '좀주', '푸',  # 우 불규칙
            '끄', '크', '트', '모으',  # ㅡ 탈락 불규칙
            '삼가', '가', '들어가',  # 거라 불규칙
            '돌아오', '오',  # 너라 불규칙
            '이르', '푸르', '누르',  # 러 불규칙
            '하',  # 여 불규칙
            '가', '노랗', '퍼렇', '놀라',  # 어미 ㄴ
            '시퍼렇', '파랗',  # ㅎ 불규칙
            '먹', '먹이',
            '보', '뵈', '뵙', '그렇',
            '좋아지', '이',  # 이었 -> 였
            '만지',  # 지 -> 져
            '서툴', '내려놓',
        }

        test_eomis = {
            '',
            '아', '어나다', '어', '워', '웠다', '워서', '왔다', '와주니', '었다', '었어', '았어', '데',
            '라', '라니까', '너라', '았다', '러', '였다', '았다', '면', '다', '거라', '고', '는', '니', '었던', '엇어', '어서',
            'ㄴ', 'ㅂ고', '운', '았다'
        }

        self.inst = Lemmatizer(stems=test_stems, endings=test_eomis)

    def __call__(self, *args, **kwargs):
        #lemmatized=[]
        #for word in args[0]:
        #    lemmatized.append(self.inst.lemmatize(word))
        return self.inst.lemmatize(args[0])

class SejongPOSLemmatizer:
    IN_TYPE = [list, tuple]
    OUT_TYPE = [list, tuple]

    def __init__(self):
        from soylemma import Lemmatizer
        self.lemmatizer = Lemmatizer(dictionary_name='default')

    def __call__(self, *args, **kwargs):
        inst = []
        for i, word_tuple in enumerate(args[0]):
            #print(str(word_tuple))
            word = word_tuple[0]
            pos = word_tuple[1]
            if str(pos).startswith('XSA+') or str(pos).startswith('XSV+'):
                lemmatized = self.lemmatizer.lemmatize(word)
                #print(str(lemmatized))
                if len(lemmatized) > 1:
                    if (lemmatized[0][0] is '히다' and lemmatized[1][0] is '한다') or (lemmatized[1][0] is '히다' and lemmatized[0][0] is '한다'):
                        if (i > 0):
                            if str(args[0][i-1][1]).startswith('N'):
                                pre_term = inst[len(inst)-1]
                                inst.remove(pre_term)
                                inst.append((pre_term[0]+lemmatized[1][0],pre_term[1]))

                    else:
                        if (i > 0):
                            if str(args[0][i-1][1]).startswith('N'):
                                pre_term = inst[len(inst)-1]
                                inst.remove(pre_term)
                                inst.append((pre_term[0]+lemmatized[0][0],pre_term[1]))

                elif len(lemmatized) == 1:
                    if (i > 0):
                        if str(args[0][i - 1][1]).startswith('N'):
                            pre_term = inst[len(inst) - 1]

                            inst.remove(pre_term)
                            inst.append((pre_term[0] + lemmatized[0][0], pre_term[1]))
            elif str(pos) == 'XSA' or str(pos) == 'XSV':
                lemmatized = self.lemmatizer.lemmatize(word)
                if len(lemmatized) > 1:
                    if (lemmatized[0][0] is '히다' and lemmatized[1][0] is '한다') or (lemmatized[1][0] is '히다' and lemmatized[0][0] is '한다'):
                        if (i > 0):
                            if str(args[0][i-1][1]).startswith('N'):
                                pre_term = inst[len(inst)-1]
                                inst.remove(pre_term)
                                inst.append((pre_term[0]+lemmatized[1][0],pre_term[1]))

                    else:
                        if (i > 0):
                            if str(args[0][i-1][1]).startswith('N'):
                                pre_term = inst[len(inst)-1]
                                inst.remove(pre_term)
                                inst.append((pre_term[0]+lemmatized[0][0],pre_term[1]))
                elif len(lemmatized) == 1:
                    if (i > 0):
                        if str(args[0][i - 1][1]).startswith('N'):
                            pre_term = inst[len(inst) - 1]
                            inst.remove(pre_term)
                            inst.append((pre_term[0] + lemmatized[0][0], pre_term[1]))
            elif str(pos).startswith('VV+') or str(pos).startswith('VA+'):
                lemmatized = self.lemmatizer.lemmatize(word)
                #(str(lemmatized))
                if len(lemmatized) > 1:
                    if (lemmatized[0][0] is '히다' and lemmatized[1][0] is '한다') or (lemmatized[1][0] is '히다' and lemmatized[0][0] is '한다'):
                        if (i > 0):
                            if str(args[0][i-1][1]).startswith('N'):
                                pre_term = inst[len(inst) - 1]
                                inst.remove(pre_term)
                                inst.append((pre_term[0]+lemmatized[1][0],pre_term[1]))
                        else:
                            inst.append((lemmatized[0][0],pos))
                    else:
                        if (i > 0):
                            if str(args[0][i-1][1]).startswith('N'):
                                pre_term = inst[len(inst)-1]
                                inst.remove(pre_term)
                                inst.append((pre_term[0]+lemmatized[0][0],pre_term[1]))
                            elif str(pos).endswith('EC'):
                                if lemmatized[0][1] == 'Verb':
                                    inst.append((lemmatized[0][0], 'VV'))
                                elif lemmatized[1][1] == 'Verb':
                                    inst.append((lemmatized[1][0], 'VV'))
                        else:
                            inst.append((lemmatized[0][0],pos))
                elif len(lemmatized) == 1:
                    if (i > 0):
                        if str(args[0][i - 1][1]).startswith('N'):
                            pre_term = inst[len(inst) - 1]
                            inst.remove(pre_term)
                            inst.append((pre_term[0] + lemmatized[0][0], pre_term[1]))
                        else:
                            inst.append((lemmatized[0][0], pos))
                    else:
                        inst.append((lemmatized[0][0], pos))
            elif str(pos).startswith('VV') or str(pos).startswith('VA'):
                if (i > 0):
                    if str(args[0][i - 1][1]) == 'VX' or str(args[0][i - 1][1]) ==  'MAG':
                        pre_term = args[0][i - 1]
                        new_term = ''
                        if pre_term[0] == '않':
                            new_term = '안'
                        else:
                            new_term = pre_term[0]

                        if new_term == '안':
                            lemmatized = new_term + ' ' + word+'다'
                            inst.append((lemmatized, pre_term[1]))
                        else:
                            lemmatized = new_term + word + '다'
                            inst.append((lemmatized, pre_term[1]))
                    elif i > 0 or i < len(args[0]):
                        lemmatized = word + '다'
                        inst.append((lemmatized, pos))
                else:
                    lemmatized = word + '다'
                    if len(lemmatized) > 0:
                        inst.append((lemmatized,pos))

            elif str(pos).startswith('VX'):
                if i > 0 and i+1 < len(args[0]):
                    if str(args[0][i + 1][1]) == 'EC' and len(inst) > 0:
                        pre_term = inst[len(inst) - 1]
                        if (pre_term[1] == 'VA'):
                            inst.remove(pre_term)
                            lemmatized = '안 ' + pre_term[0]
                            inst.append((lemmatized, pos))
                        else:
                            lemmatized = word + '다'
                            inst.append((lemmatized, pos))


            elif (pos != 'EC' and pos != 'JX'):
                inst.append((word,pos))

        #print(inst)
        return inst
