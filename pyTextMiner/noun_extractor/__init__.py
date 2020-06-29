from soynlp.noun import LRNounExtractor_v2

class BaseNounExtraction:
    INPUT=[str]
    OUTPUT=[list,str]

class NounExtractionKorean(BaseNounExtraction):
    def __init__(self,sents):
        self.inst = LRNounExtractor_v2(verbose=False, extract_compound=True)
        self.inst.train(sents)
        self.inst.extract()

    def __call__(self, *args, **kwargs):
        return self.inst.decompose_compound(args[0])