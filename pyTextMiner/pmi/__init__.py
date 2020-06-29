
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.vectorizer import sent_to_word_contexts_matrix
from soynlp.word import pmi as pmi_func

class BasePMICalculator:
    INPUT=[str]
    OUTPUT=[list,tuple]

class PMICalculator(BasePMICalculator):
    def __init__(self, corpus=None):
        word_extractor = WordExtractor()
        word_extractor.train(corpus)
        cohesions = word_extractor.all_cohesion_scores()
        l_cohesions = {word: score[0] for word, score in cohesions.items()}
        tokenizer = LTokenizer(l_cohesions)
        x, self.idx2vocab = sent_to_word_contexts_matrix(
            corpus,
            windows=3,
            min_tf=10,
            tokenizer=tokenizer,  # (default) lambda x:x.split(),
            dynamic_weight=False,
            verbose=True)

        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}

        self.pmi, px, py = pmi_func(
            x,
            min_pmi=0,
            alpha=0.0,
            beta=0.75
        )
    def __call__(self, *args, **kwargs):
        query = self.vocab2idx[args[0]]
        submatrix = self.pmi[query, :].tocsr()  # get the row of query
        contexts = submatrix.nonzero()[1]  # nonzero() return (rows, columns)
        pmi_i = submatrix.data

        most_relateds = [(idx, pmi_ij) for idx, pmi_ij in zip(contexts, pmi_i)]
        most_relateds = sorted(most_relateds, key=lambda x: -x[1])[:10]
        most_relateds = [(self.idx2vocab[idx], pmi_ij) for idx, pmi_ij in most_relateds]

        return most_relateds