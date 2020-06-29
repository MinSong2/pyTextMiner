import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

class Cooccurrence(CountVectorizer):
    """Co-ocurrence matrix
    Convert collection of raw documents to word-word co-ocurrence matrix

    Parameters
    ----------
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    max_df: float in range [0, 1] or int, default=1.0

    min_df: float in range [0, 1] or int, default=1

    Example
    -------

    >> import Cooccurrence
    >> docs = ['this book is good',
               'this cat is good',
               'cat is good shit']
    >> model = Cooccurrence()
    >> Xc = model.fit_transform(docs)

    Check vocabulary by printing
    >> model.vocabulary_

    """

    def __init__(self, encoding='utf-8', ngram_range=(1, 1),
                 max_df=1.0, min_df=1, max_features=None,
                 stop_words=None, normalize=True, vocabulary=None):

        super(Cooccurrence, self).__init__(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words=stop_words,
            vocabulary=vocabulary
        )

        self.X = None

        self.normalize = normalize

    def fit_transform(self, raw_documents, y=None):
        """Fit cooccurrence matrix

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        Xc : Cooccurrence matrix

        """
        X = super(Cooccurrence, self).fit_transform(raw_documents)
        self.X = X

        n_samples, n_features = X.shape

        Xc = (X.T * X)
        if self.normalize:
            g = sp.diags(1./Xc.diagonal())
            Xc = g * Xc
        else:
            Xc.setdiag(0)

        return Xc

    def vocab(self):
        tuples = super(Cooccurrence, self).get_feature_names()
        vocabulary=[]
        for e_tuple in tuples:
            tokens = e_tuple.split()
            for t in tokens:
                if t not in vocabulary:
                    vocabulary.append(t)

        return vocabulary

    def word_histgram(self):
        word_list = super(Cooccurrence, self).get_feature_names()
        count_list = self.X.toarray().sum(axis=0)
        return dict(zip(word_list,count_list))