''' list(str) => list(tuple) '''

class BaseTagger:
    IN_TYPE = [list, str]
    OUT_TYPE = [list, tuple]

class NLTK(BaseTagger):
    def __init__(self):
        import nltk
        nltk.download('averaged_perceptron_tagger')

        from nltk.tag.perceptron import PerceptronTagger

        self.inst = PerceptronTagger()

    def __call__(self, *args, **kwargs):
        return self.inst.tag(args[0])