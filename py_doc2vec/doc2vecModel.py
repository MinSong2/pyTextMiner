"""Train a Doc2vec model from a collection of documents."""

import os
import sys
import codecs
import argparse
import logging
import shutil
import json
from random import shuffle, randint
from datetime import datetime
from collections import namedtuple, OrderedDict
import multiprocessing
import gensim
from scipy import spatial
from gensim.models import Doc2Vec
import time


class Doc2VecTrainer:
    def __init__(self):
        print('Doc2VecTrainer')
        self.TaggedDocument = namedtuple('TaggedDocument', 'tags words')

    def read_lines(self, path):
        '''Return lines in file'''
        return [line.strip() for line in codecs.open(path, "r", "utf-8")]

    def current_time_ms(self):
        return int(time.time() * 1000.0)

    def clean_make_dir(self, path):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)

    def make_timestamped_dir(self, base_path, algorithm, vector_size, epochs, window):
        suffix = '%s_dim=%d_window=%d_epochs=%d' % (algorithm, vector_size, window, epochs)
        output_path = os.path.join(base_path, str(self.current_time_ms())) + '_' + suffix
        self.clean_make_dir(output_path)
        return output_path

    def load_stopwords(self, stopwords_path):
        logging.info("Loading stopwords: %s", stopwords_path)
        stopwords = self.read_lines(stopwords_path)
        return dict(map(lambda w: (w.lower(), ''), stopwords))

    def run(self, documents, output_base_dir, vocab_min_count, num_epochs, algorithm, vector_size, alpha,
            min_alpha, train, window, cores):

        # As soon as FAST_VERSION is not -1, there are compute-intensive codepaths that avoid holding
        # the python global interpreter lock, and thus you should start to see multiple cores engaged.
        # For more details see: https://github.com/RaRe-Technologies/gensim/issues/532
        # assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

        if cores == None:
            cores = multiprocessing.cpu_count()

        negative = 5
        hs = 0

        if algorithm == 'pv_dmc':
            # PV-DM with concatenation
            # window=5 (both sides) approximates paper's 10-word total window size
            # PV-DM w/ concatenation adds a special null token to the vocabulary: '\x00'
            model = Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                            min_count=vocab_min_count, workers=cores)
        elif algorithm == 'pv_dma':
            # PV-DM with average
            # window=5 (both sides) approximates paper's 10-word total window size
            model = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                            min_count=vocab_min_count, workers=cores)
        elif algorithm == 'pv_dbow':
            # PV-DBOW
            model = Doc2Vec(dm=0, vector_size=vector_size, window=window, negative=negative, hs=hs,
                            min_count=vocab_min_count, workers=cores)
        else:
            raise ValueError('Unknown algorithm: %s' % algorithm)

        logging.info('Algorithm: %s' % str(model))

        logging.info('Build vocabulary')
        model.build_vocab(documents)
        vocab_size = len(model.wv.vocab)
        logging.info('Vocabulary size: %d', vocab_size)

        target_dir = self.make_timestamped_dir(output_base_dir, algorithm, model.vector_size, num_epochs, window)
        vocab_path = os.path.join(target_dir, 'vocabulary')
        logging.info('Save vocabulary to: %s', vocab_path)
        with open(vocab_path, 'w') as f:
            term_counts = [[term, value.count] for term, value in model.wv.vocab.items()]
            term_counts.sort(key=lambda x: -x[1])
            for x in term_counts:
                f.write('%s, %d\n' % (x[0], x[1]))

        if train:
            logging.info('Shuffle documents')
            shuffle(documents)

            logging.info('Train model')
            model.train(documents, total_examples=len(documents), epochs=num_epochs, start_alpha=alpha,
                        end_alpha=min_alpha)

            logging.info('Save model to: %s', target_dir)
            model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
            model.save(os.path.join(target_dir, 'doc2vec.model'))

            model_meta = {
                'argv': sys.argv,
                'target_dir': target_dir,
                'algorithm': algorithm,
                'window': window,
                'vector_size': vector_size,
                'alpha': alpha,
                'min_alpha': min_alpha,
                'num_epochs': num_epochs,
                'vocab_min_count': vocab_min_count,
                'vocab_size': vocab_size,
                'cores': cores,
                'negative': negative,
                'hs': hs
            }

            model_meta_path = os.path.join(target_dir, 'model.meta')
            logging.info('Save model metadata to: %s', model_meta_path)
            with open(model_meta_path, 'w') as outfile:
                json.dump(model_meta, outfile)


class Doc2VecSimilarity:
    def __init__(self):
        print('doc2vec based similarity')
        self.doc2vec = None

    def load_model(self, model_file):
        self.doc2vec = Doc2Vec.load(model_file)

    def most_similar(self, document):
        print('most similart documents')
        doc_vec = self.doc2vec.infer_vector(document.split())
        similars = self.doc2vec.docvecs.most_similar(positive=[doc_vec])
        return similars

    def most_similar_vec(self, document_vec):
        print('most similart documents')
        doc_vec = self.doc2vec.infer_vector(document_vec)
        similars = self.doc2vec.docvecs.most_similar(positive=[doc_vec])
        return similars

    def compute_similarity(self, first_document, second_document):
        vec1 = self.doc2vec.infer_vector(first_document.split(), steps=50, alpha=0.25)
        vec2 = self.doc2vec.infer_vector(second_document.split(), steps=50, alpha=0.25)

        similarity = spatial.distance.cosine(vec1, vec2)
        return similarity

    def compute_similarity_vec(self, first_vec=[], second_vec=[]):
        vec1 = self.doc2vec.infer_vector(first_vec, steps=50, alpha=0.25)
        vec2 = self.doc2vec.infer_vector(second_vec, steps=50, alpha=0.25)

        similarity = spatial.distance.cosine(vec1, vec2)
        return similarity
