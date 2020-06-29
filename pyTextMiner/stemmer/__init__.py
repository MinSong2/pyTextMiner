''' str => str '''

class BaseStemmer:
    IN_TYPE = [str]
    OUT_TYPE = [str]

class Porter(BaseStemmer):
    def __init__(self):
        import nltk
        self.inst = nltk.stem.PorterStemmer()

    def __call__(self, *args, **kwargs):
        return self.inst.stem(args[0])

class Lancaster(BaseStemmer):
    def __init__(self):
        import nltk
        self.inst = nltk.stem.LancasterStemmer()

    def __call__(self, *args, **kwargs):
        return self.inst.stem(args[0])