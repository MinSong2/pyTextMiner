import string
from collections import Counter
import os
from nltk import bigrams
from collections import defaultdict
import operator
import numpy as np

class BaseCooccurrence:
    INPUT=[list,str]
    OUTPUT=[list,tuple]

class CooccurrenceWorker(BaseCooccurrence):
    def __init__(self):
        name = 'cooccurrence'

        from sklearn.feature_extraction.text import CountVectorizer
        import pyTextMiner.cooccurrence.cooccurrence as co
        self.inst = co.Cooccurrence(ngram_range=(2, 2), stop_words='english')

    def __call__(self, *args, **kwargs):

        # bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary={'awesome unicorns': 0, 'batman forever': 1})
        co_occurrences = self.inst.fit_transform(args[0])
        # print('Printing sparse matrix:', co_occurrences)
        # print(co_occurrences.todense())
        sum_occ = np.sum(co_occurrences.todense(), axis=0)
        # print('Sum of word-word occurrences:', sum_occ)

        # Converting itertor to set
        result = zip(self.inst.get_feature_names(), np.array(sum_occ)[0].tolist())
        result_set = list(result)
        return result_set, self.inst.vocab()

class CooccurrenceManager:
    def __init__(self):
        self.d = {}  # 단어->단어ID로 변환할때 사용
        self.w = []  # 단어ID->단어로 변환할 때 사용

    def getIdOrAdd(self, word):
        # 단어가 이미 사전에 등록된 것이면 해당하는 ID를 돌려주고
        if word in self.d: return self.d[word]
        # 그렇지 않으면 새로 사전에 등록하고 ID를 부여함
        self.d[word] = len(self.d)
        self.w.append(word)
        return len(self.d) - 1

    def getWord(self, id):
        return self.w[id]

    def calculateCooccurrence(self, list):
        count = {}  # 동시출현 빈도가 저장될 dict
        words = list(set(list))  # 단어별로 분리한 것을 set에 넣어 중복 제거하고, 다시 list로 변경
        wids = [self.getIdOrAdd(w) for w in words]
        for i, a in enumerate(wids):
            for b in wids[i + 1:]:
                if a == b: continue  # 같은 단어의 경우는 세지 않음
                if a > b: a, b = b, a  # A, B와 B, A가 다르게 세어지는것을 막기 위해 항상 a < b로 순서 고정
                count[a, b] = count.get((a, b), 0) + 1  # 실제로 센다

        sorted = []
        for tup in count:
            freq = count[tup]
            left_word = self.getWord(count[0])
            right_word = self.getWord(count[1])
            sorted.append(((left_word, right_word), freq))
        return sorted, words

    def computeCooccurence(self, list, target=''):
        com = defaultdict(lambda: defaultdict(int))
        count_all = Counter()
        count_all1 = Counter()

        uniqueList = []
        for _array in list:
            for line in _array:
                for word in line:
                    if len(target) < 1:
                        if word not in uniqueList:
                            uniqueList.append(word)

                terms_bigram = bigrams(line)
                # Update the counter
                count_all.update(line)
                count_all1.update(terms_bigram)

                # Build co-occurrence matrix
                for i in range(len(line) - 1):
                    for j in range(i + 1, len(line)):
                        w1, w2 = sorted([line[i], line[j]])
                        if w1 != w2:
                            com[w1][w2] += 1



        com_max = []
        # For each term, look for the most common co-occurrent terms
        for t1 in com:
            t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
            for t2, t2_count in t1_max_terms:
                if len(target)>0 and (target is t1 or target is t2):
                    if t1 not in uniqueList:
                        uniqueList.append(t1)
                    if t2 not in uniqueList:
                        uniqueList.append(t2)
                    com_max.append(((t1, t2), t2_count))
        # Get the most frequent co-occurrences
        terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)

        return terms_max, uniqueList

