from itertools import chain

class BaseNgram:
    IN_TYPE = [list, str]
    OUT_TYPE = [list, str]

class NGramTokenizer(BaseNgram):
    def __init__(self, min=1, ngramCount=3, concat='_'):
        self.ngramCount = ngramCount
        self.min = min
        self.converted = []
        self.concat = concat

    def __call__(self, *args, **kwargs):
        converted = []
        from nltk.util import ngrams
        for i in range(self.min, self.ngramCount+1):
            output = list(ngrams((args[0]), i))
            for x in output:
                if (len(x) > 0):
                    converted.append(self.concat.join(x))

        #print("NGRAM " + str(converted))
        self.converted = converted

        return converted