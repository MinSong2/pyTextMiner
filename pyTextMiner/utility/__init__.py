
from gensim.models import fasttext
from soynlp.hangle import decompose, compose
import re

class Utility:
    def __init__(self):
        name = 'Utility Class'
        self.doublespace_pattern = re.compile('\s+')

    def jamo_sentence(self, sent):
        def transform(char):
            if char == ' ':
                return char
            cjj = decompose(char)
            if cjj != None:
                if len(cjj) == 1:
                    return cjj
                cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)

            if cjj == None:
                return ''

            return cjj_

        sent_ = ''.join(transform(char) for char in sent)
        sent_ = self.doublespace_pattern.sub(' ', sent_)
        return sent_

    def decode(self, s):
        def process(t):
            assert len(t) % 3 == 0
            t_ = t.replace('-', ' ')
            chars = [tuple(t_[3 * i:3 * (i + 1)]) for i in range(len(t_) // 3)]
            recovered = []
            for char in chars:
                try:
                    composed = compose(*char)
                    recovered.append(composed)
                except:
                    pass
            #recovered = [compose(*char) for char in chars]
            recovered = ''.join(recovered)
            return recovered

        return ' '.join(process(t) for t in s.split())

    def decode_sentence(self, sent):
        return ' '.join(self.decode(token) for token in sent.split())

    def cosine_similarity(self, word1, word2, model):
        cjj1 = self.jamo_sentence(word1)
        cjj2 = self.jamo_sentence(word2)
        cos_sim = model.cosine_similarity(cjj1, cjj2)
        return cos_sim

    def most_similar(self, word, model):
        jamo_result = []
        cjj = self.jamo_sentence(word)
        result = model.most_similar(cjj)
        for token in result:
            word = token[0]
            encoded_word = self.decode(word)
            sim = token[1]
            jamo_result.append((encoded_word,sim))

        return jamo_result

    def most_similars(self, model, positives, negatives, topn=10):
        jamo_result = []
        result = model.most_similar(positive=positives,negative=negatives,topn=topn)
        for token in result:
            word = token[0]
            if len(word) > 3:
                encoded_word = self.decode(word)
                sim = token[1]
                jamo_result.append((encoded_word,sim))

        return jamo_result

    def similar_by_word(self, model, word, topn=10):
        jamo_result = []
        result = model.similar_by_word(word, topn=topn)
        for token in result:
            word = token[0]
            if len(word) > 3:
                encoded_word = self.decode(word)
                sim = token[1]
                jamo_result.append((encoded_word, sim))

        return jamo_result