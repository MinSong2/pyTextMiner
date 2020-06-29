''' str => list(str) '''

class BaseSplitter:
    IN_TYPE = [str]
    OUT_TYPE = [list, str]

class SpecialCharRemover(BaseSplitter):
    IN_TYPE = [str]
    OUT_TYPE = [str]

    def __init__(self):
        import re
        self.hangul = re.compile('[^ ㄱ-ㅣ가-힣\\.\\?\\,]+')

    def __call__(self, *args, **kwargs):
        return self.hangul.sub('', *args)

class NLTK(BaseSplitter):
    def __init__(self):
        import nltk.tokenize
        self.func = nltk.tokenize.sent_tokenize

    def __call__(self, *args, **kwargs):
        return self.func(*args)

class KoSentSplitter(BaseSplitter):
    def __init__(self):
        import kss
        self.func = kss.split_sentences

    def __call__(self, *args, **kwargs):
        return self.func(*args)
