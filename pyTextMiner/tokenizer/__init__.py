''' str => list(str) '''

class BaseTokenizer:
    IN_TYPE = [str]
    OUT_TYPE = [list, str]

# [English]
class Tweet(BaseTokenizer):
    def __init__(self):
        import nltk.tokenize
        self.inst = nltk.tokenize.TweetTokenizer()

    def __call__(self, *args, **kwargs):
        return self.inst.tokenize(*args)

class Whitespace(BaseTokenizer):
    def __init__(self):
        import nltk.tokenize
        self.inst = nltk.tokenize.WhitespaceTokenizer()

    def __call__(self, *args, **kwargs):
        return self.inst.tokenize(*args)

class Word(BaseTokenizer):
    def __init__(self):
        import nltk.tokenize
        self.inst = nltk.tokenize.word_tokenize

    def __call__(self, *args, **kwargs):
        print(str(self.inst(*args)))
        return self.inst(*args)

class WordPos(BaseTokenizer):
    def __init__(self):
        import nltk
        self.inst = nltk
        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        tokens = self.inst.word_tokenize(*args)

        return self.inst.pos_tag(tokens)

# [Korean]
class Komoran(BaseTokenizer):
    def __init__(self,userdic=None):
        from konlpy.tag import Komoran
        import os
        if userdic is not None:
            print("user dict " + str(os.path.abspath(userdic)))
            self.inst = Komoran(userdic=os.path.abspath(userdic))
        else:
            self.inst = Komoran()
        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        return self.inst.pos(args[0])

class TwitterKorean(BaseTokenizer):
    def __init__(self):
        from konlpy.tag import Twitter
        self.inst = Twitter()

        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        return self.inst.pos(args[0])

class KokomaKorean(BaseTokenizer):
    def __init__(self):
        from konlpy.tag import Kkma
        self.inst = Kkma()

        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        return self.inst.pos(args[0])

class MeCab(BaseTokenizer):
    def __init__(self, path=None):
        #import MeCab
        #self.inst = MeCab.Tagger()

        from konlpy.tag import Mecab
        self.inst = Mecab(path)

        self.OUT_TYPE = [list, tuple]

    def __call__(self, *args, **kwargs):
        try:
            return self.inst.pos(args[0])
        except:
            return []

class SpecialTokenizer:
    IN_TYPE = [str]
    OUT_TYPE = [str]

class MaxScoreTokenizerKorean(SpecialTokenizer):
    def __init__(self, scores=None):
        from soynlp.tokenizer import MaxScoreTokenizer
        self.inst=MaxScoreTokenizer(scores=scores)
        self.OUT_TYPE = [list, str]

    def __call__(self, *args, **kwargs):
        tokens = self.inst.tokenize(args[0])
        return tokens

class LTokenizerKorean(SpecialTokenizer):
    def __init__(self, scores=None):
        from soynlp.tokenizer import LTokenizer
        self.inst=LTokenizer(scores=scores)

        self.OUT_TYPE = [list, str]

    def __call__(self, *args, **kwargs):
        tokens = self.inst.tokenize(args[0])
        return tokens

class RegexTokenizerKorean(SpecialTokenizer):
    def __init__(self):
        from soynlp.tokenizer import RegexTokenizer
        self.inst=RegexTokenizer()
        self.OUT_TYPE = [list, str]

    def __call__(self, *args, **kwargs):
        tokens=self.inst.tokenize(args[0])
        return tokens