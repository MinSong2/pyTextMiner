class BaseChunker:
    IN_TYPE = [str]
    OUT_TYPE = [list, str]

class KoreanChunker(BaseChunker):
    def __init__(self):

        import nltk
        grammar = """
        NP: {<N.*>*<Suffix>?}   # Noun phrase
        VP: {<V.*>*}            # Verb phrase
        AP: {<A.*>*}            # Adjective phrase
        """

        self.inst=nltk.RegexpParser(grammar)


    def __call__(self, *args, **kwargs):
        import konlpy
        words = konlpy.tag.Komoran().pos(*args)

        chunks = self.inst.parse(words)

        return chunks