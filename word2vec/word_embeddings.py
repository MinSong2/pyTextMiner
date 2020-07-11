import io
import array
import collections
import numpy as np
import multiprocessing
from time import time
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import FastText

import pyTextMiner as ptm
from gensim.models import Word2Vec


class WordEmbeddings:
    def __init__(self):
        self.documents = []

    def preprocessing(self, mode, path, stopword_file, files, is_directory=False, doc_index=-1, max=-1):
        util = ptm.Utility()
        # mode is either filtered or unfiltered or simple
        corpus = []
        if mode == 'unfiltered':
            # path = '/usr/local/lib/mecab/dic/mecab-ko-dic'
            pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                                    ptm.tokenizer.MeCab(path),
                                    ptm.lemmatizer.SejongPOSLemmatizer(),
                                    ptm.helper.SelectWordOnly(),
                                    ptm.helper.StopwordFilter(file=stopword_file))

            for a_file in files:
                if is_directory == True and max == -1:
                    corpus += ptm.CorpusFromDirectory(a_file).docs
                elif is_directory == False and doc_index != -1 and max == -1:
                    corpus += ptm.CorpusFromFieldDelimitedFile(a_file, doc_index).docs
                elif is_directory == False and doc_index == -1 and max == -1:
                    corpus += ptm.CorpusFromFile(a_file).docs
                elif is_directory == False and max > 0:
                    count = 0
                    docs = []
                    for line in open(a_file):
                        if doc_index != -1:
                            line = line.split()[doc_index]
                        if len(line) < 1:
                            continue
                        toks = line.split()
                        if len(toks) > 10:
                            docs.append(line)
                            count += 1
                        if count % 10000 == 0:
                            print('processing... ' + str(count))
                        if max < count:
                            break

                        corpus = ptm.Corpus(docs)

            if type(corpus) != list and len(corpus.docs) > 0 or type(corpus) == list and len(corpus) > 0:
                result = pipeline.processCorpus(corpus)
                for doc in result:
                    document = []
                    for sent in doc:
                        for word in sent:
                            document.append(word)
                    self.documents.append(document)

        elif mode == 'filtered':
            pipeline = ptm.Pipeline(ptm.tokenizer.Word())
            # corpus = ptm.CorpusFromFile('/Data/ko_sns_comments/naver_comments15_16_filtered.txt')
            for a_file in files:
                if is_directory == True and max == -1:
                    corpus += ptm.CorpusFromDirectory(a_file).docs
                elif is_directory == False and doc_index != -1 and max == -1:
                    corpus += ptm.CorpusFromFieldDelimitedFile(a_file, doc_index).docs
                elif is_directory == False and doc_index == -1 and max == -1:
                    corpus += ptm.CorpusFromFile(a_file).docs
                elif is_directory == False and max > 0:
                    count = 0
                    docs = []
                    for line in open(a_file):
                        if doc_index != -1:
                            line = line.split()[doc_index]
                        if len(line) < 1:
                            continue
                        toks = line.split()
                        if len(toks) > 10:
                            docs.append(line)
                            count += 1
                        if count % 10000 == 0:
                            print('processing... ' + str(count))
                        if max < count:
                            break
                        corpus = ptm.Corpus(docs)

            self.documents = pipeline.processCorpus(corpus)

        elif mode == 'jamo_split_unfiltered':
            # documents = LineSentence(datapath('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'))
            pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                                    ptm.tokenizer.MeCab(path),
                                    ptm.lemmatizer.SejongPOSLemmatizer(),
                                    ptm.helper.SelectWordOnly(),
                                    ptm.helper.StopwordFilter(file=stopword_file))

            for a_file in files:
                if is_directory == True and max == -1:
                    corpus += ptm.CorpusFromDirectory(a_file).docs
                elif is_directory == False and doc_index != -1 and max == -1:
                    corpus += ptm.CorpusFromFieldDelimitedFile(a_file, doc_index).docs
                elif is_directory == False and doc_index == -1 and max == -1:
                    corpus += ptm.CorpusFromFile(a_file).docs
                elif is_directory == False and max > 0:
                    count = 0
                    docs = []
                    for line in open(a_file):
                        if doc_index != -1:
                            line = line.split()[doc_index]
                        if len(line) < 1:
                            continue
                        toks = line.split()
                        if len(toks) > 10:
                            docs.append(line)
                            count += 1
                        if count % 10000 == 0:
                            print('processing... ' + str(count))
                        if max < count:
                            break

                        corpus = ptm.Corpus(docs)

            if type(corpus) != list and len(corpus.docs) > 0 or type(corpus) == list and len(corpus) > 0:
                result = pipeline.processCorpus(corpus)
                for doc in result:
                    for sent in doc:
                        _sent = ''
                        for word in sent:
                            _sent += word + ' '
                        _sent = _sent.strip()
                        _sent = util.jamo_sentence(_sent)
                        toks = _sent.split()
                        if len(toks) > 10:
                            self.documents.append(toks)

        elif mode == 'jamo_split_filtered':
            # documents = LineSentence(datapath('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'))
            pipeline = ptm.Pipeline(ptm.tokenizer.Word())
            for a_file in files:
                if is_directory == True and max == -1:
                    corpus += ptm.CorpusFromDirectory(a_file).docs
                elif is_directory == False and doc_index != -1 and max == -1:
                    corpus += ptm.CorpusFromFieldDelimitedFile(a_file, doc_index)
                elif is_directory == False and doc_index == -1 and max == -1:
                    corpus += ptm.CorpusFromFile(a_file)
                elif is_directory == False and max > 0:
                    count = 0
                    docs = []
                    for line in open(a_file):
                        if doc_index != -1:
                            line = line.split()[doc_index]
                        if len(line) < 1:
                            continue
                        toks = line.split()
                        if len(toks) > 10:
                            docs.append(line)
                            count += 1
                        if count % 10000 == 0:
                            print('processing... ' + str(count))
                        if max < count:
                            break

                        corpus = ptm.Corpus(docs)

            if type(corpus) != list and len(corpus.docs) > 0 or type(corpus) == list and len(corpus) > 0:
                result = pipeline.processCorpus(corpus)
                for doc in result:
                    _sent = ''
                    for word in doc:
                        _sent += word + ' '
                    _sent = _sent.strip()
                    _sent = util.jamo_sentence(_sent)
                    toks = _sent.split()
                    if len(toks) > 10:
                        self.documents.append(toks)

        elif mode == 'simple':
            # documents = LineSentence(datapath('/Data/ko_sns_comments/naver_comments15_16_filtered.txt'))
            count = 0
            for line in open(files[0]):
                if doc_index != -1:
                    line = line.split()[doc_index]
                toks = line.split()
                if len(toks) > 10:
                    self.documents.append(toks)
                    count += 1

                if count % 10000 == 0:
                    print('processing... ' + str(count))

        print('Document size for the total dataset: ' + str(len(self.documents)))

    def train(self, min_count=5, window=5, size=300, negative=20):
        t = time()

        cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
        print('default train function')
        self.model = gensim.models.Word2Vec(min_count=min_count,
                                            window=window,
                                            size=size,
                                            sample=6e-5,
                                            alpha=0.03,
                                            min_alpha=0.0007,
                                            negative=negative,
                                            workers=cores - 1)

        self.model.build_vocab(self.documents, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        self.model.train(self.documents, total_examples=self.model.corpus_count, epochs=30, report_delay=1)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    def save_model(self, model_file, binary=True):
        self.model.wv.save_word2vec_format(model_file, binary=binary)

    def load_model(self, model_file, binary=True):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=binary, unicode_errors='ignore')

    def most_similar(self, positives, negatives, topn=10):
        return self.model.most_similar(positive=positives, negative=negatives, topn=topn)

    def similar_by_word(self, word):
        return self.model.similar_by_word(word)


class Word2Vec(WordEmbeddings):
    def __init__(self):
        print('Word2Vec')
        super().__init__()

    def preprocessing(self, mode, mecab_path, stopword_file, files, is_directory=False, doc_index=-1, max=-1):
        super().preprocessing(mode, mecab_path, stopword_file, files, is_directory, doc_index, max)

    def train(self, min_count=5, window=5, size=300, negative=20):
        super().train(min_count, window, size, negative)

    def save_model(self, model_file, binary=True):
        super().save_model(model_file, binary)

    def load_model(self, model_file, binary=True):
        super().load_model(model_file, binary)

    def most_similar(self, positives, negatives, topn=10):
        return super().most_similar(positives, negatives, topn)

    def similar_by_word(self, word):
        return super().similar_by_word(word)


class FastText(WordEmbeddings):
    def __init__(self):
        self.model = None
        print('FastText')
        super().__init__()

    def preprocessing(self, mode, mecab_path, stopword_file, files, is_directory=False, doc_index=-1, max=-1):
        super().preprocessing(mode, mecab_path, stopword_file, files, is_directory, doc_index, max)

    def train(self, min_count=5, window=5, size=300, negative=20, sg=0, min_n=2, max_n=6):
        print('min ' + str(min_count))
        self.model = gensim.models.FastText(self.documents, size=size, window=window,
                                            negative=negative, min_count=min_count, sg=sg, min_n=min_n, max_n=max_n,
                                            workers=10)

    def save_model(self, model_file):
        self.model.save(model_file)

    def load_model(self, model_file):
        from gensim.models.fasttext import FastText as gensim_ft
        self.model = gensim_ft.load(model_file)

    def most_similar(self, mode, positives, negatives, topn=10):
        # mode = 'jamo_split'
        similarities = []
        if mode != 'jamo_split':
            similarities = self.model.most_similar(positive=positives, negative=negatives, topn=topn)
        else:
            util = ptm.Utility()
            jamo_positives = []
            for positive in positives:
                jamo_positives.append(util.jamo_sentence(positive))
            jamo_negatives = []
            for negative in negatives:
                jamo_negatives.append(util.jamo_sentence(negative))
            similarities = util.most_similars(self.model, positives=jamo_positives, negatives=jamo_negatives, topn=topn)

        return similarities

    def similar_by_word(self, mode, word):
        if mode != 'jamo_split':
            return self.model.similar_by_word(word)
        else:
            util = ptm.Utility()
            return util.similar_by_word(self.model, util.jamo_sentence(word))


class GloVe(WordEmbeddings):
    import numpy as np
    import io
    def __init__(self):
        print('GloVe')
        super().__init__()

    def preprocessing(self):
        print('not implemented')

    def train(self):
        print('not implemented')

    def load_model(self, model_file):
        dct = {}
        vectors = array.array('d')

        # Read in the data.
        with io.open(model_file, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                tokens = line.split(' ')

                word = tokens[0]
                entries = tokens[1:]

                dct[word] = i
                vectors.extend(float(x) for x in entries)

        # Infer word vectors dimensions.
        no_components = len(entries)
        no_vectors = len(dct)

        # Set up the model instance.
        self.no_components = no_components
        self.word_vectors = (np.array(vectors)
                             .reshape(no_vectors,
                                      no_components))
        self.word_biases = np.zeros(no_vectors)
        self.add_dictionary(dct)

    def add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()

        self.inverse_dictionary = {v: k for k, v in items_iterator}

    def _similarity_query(self, word_vec, number):

        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
                if x in self.inverse_dictionary]

    def most_similar(self, word, topn=10):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        return self._similarity_query(self.word_vectors[word_idx], topn)[1:]

    def most_similars(self, positives, negatives, topn=10):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            #print(str(self.word_vectors.shape))
            embeddings = np.zeros(self.word_vectors.shape, "float32")
            idx = 0
            for i in positives[::2]:
                if len(positives) == 1:
                    word_idx = self.dictionary[i]
                    embeddings = self.word_vectors[word_idx]
                else:
                    j = positives[idx+1]
                    word_idx1 = self.dictionary[i]
                    word_idx2 = self.dictionary[j]
                    embeddings = np.add(self.word_vectors[word_idx1], self.word_vectors[word_idx2])
                    idx += 2;
            for i in negatives:
                word_idx = self.dictionary[i]
                embeddings = np.subtract(embeddings, self.word_vectors[word_idx])

        except KeyError:
            raise Exception('Word not in dictionary')

        return self._similarity_query(embeddings, topn)[1:]