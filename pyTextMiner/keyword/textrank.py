import nltk
from nltk.tokenize import sent_tokenize
from konlpy.tag import Mecab
import networkx as nx
import math
import matplotlib.pyplot as plt
import numpy as np
import re

WINDOW_SIZE = 10
STOP_POS = {
    'ko': ['IC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JC', 'JX', 'XR', 'SF', 'SE', 'SSO', 'SSC', 'SC',
           'SY', 'EC', 'EF', 'ETN', 'ETM', 'XSV', 'XSA', 'XSN', 'XPN', 'NP'],
    'en': ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP', 'RB', 'RBR',
           'RBS', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP', 'WRB', 'SL', 'SY', 'SN', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
           'VBZ']}  # 'NN','NNS','NNP','NNPS',


class TextRank:
    def __init__(self, pos_tagger_name='mecab', mecab_path='',
                 exceptional_stop_pos=[], lang='ko', stopwords=[]):
        self.stop_pos = STOP_POS[lang]
        # print(self.stop_pos)

        if pos_tagger_name == 'mecab':
            from konlpy.tag import Mecab
            self.pos_tagger = Mecab(mecab_path)
        elif pos_tagger_name == 'komoran':
            from konlpy.tag import Komoran
            self.pos_tagger = Komoran()
        elif pos_tagger_name == 'nltk':
            self.pos_tagger = None
        else:
            from konlpy.tag import Okt
            self.pos_tagger = Okt()

        if not exceptional_stop_pos:
            self.stop_pos = [x for x in self.stop_pos if x not in exceptional_stop_pos]

        self.stopwords = []
        if stopwords:
            self.stopwords = stopwords

        self.graph = nx.diamond_graph()
        self.graph.clear()  # 처음 생성시 graph에 garbage node가 남아있어 삭제

        self.tokens = []
    def keywords(self, text, n=10, window_size=WINDOW_SIZE):
        if self.pos_tagger is None:
            import nltk
            _tokens = nltk.word_tokenize(text)
            tokenized = nltk.pos_tag(_tokens)
        else:
            tokenized = self.pos_tagger.pos(text)
        # print(tokenized)

        nodes = []
        tokens = []
        for token in tokenized:
            if (len(token[0]) > 1) & (str(token[0]) not in self.stopwords) & (token[1][:3] not in self.stop_pos) & (
                    token[0] != token[1]):  # nltk는 특수문자는 품사가 태깅이 안되고 토큰이 그대로 품사에 나옴
                nodes.append(token)
            tokens.append(token)

        def connect(nodes, tokens):
            edges = []
            for window_start in range(0, (len(tokens) - window_size + 1)):
                window = tokens[window_start:window_start + window_size]

                for i in range(window_size):
                    for j in range(window_size):
                        if (i > j) & (window[i] in nodes) & (window[j] in nodes):
                            edges.append((window[i], window[j]))
            return edges

        graph = nx.diamond_graph()
        graph.clear()  # 처음 생성시 graph에 garbage node가 남아있어 삭제
        graph.add_nodes_from(list(set(nodes)))  # node 등록
        graph.add_edges_from(connect(nodes, tokens))  # edge 연결
        scores = nx.pagerank(graph)  # pagerank 계산
        rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # score 역순 정렬
        return rank[:n]

    def build_keywords(self, text, window_size=WINDOW_SIZE):
        if self.pos_tagger is None:
            import nltk
            _tokens = nltk.word_tokenize(text)
            tokenized = nltk.pos_tag(_tokens)
        else:
            tokenized = self.pos_tagger.pos(text)

        # print(tokenized)
        nodes = []
        tokens = []
        for token in tokenized:
            if (len(token[0]) > 1) & (str(token[0]) not in self.stopwords) & (token[1][:3] not in self.stop_pos) & (
                    token[0] != token[1]):  # nltk는 특수문자는 품사가 태깅이 안되고 토큰이 그대로 품사에 나옴
                nodes.append(token)
            tokens.append(token)
            self.tokens.append(token[0])

        def connect(nodes, tokens):
            edges = []
            for window_start in range(0, (len(tokens) - window_size + 1)):
                window = tokens[window_start:window_start + window_size]

                for i in range(window_size):
                    for j in range(window_size):
                        if (i > j) & (window[i] in nodes) & (window[j] in nodes):
                            edges.append((window[i], window[j]))
            return edges

        self.graph.add_nodes_from(list(set(nodes)))  # node 등록
        self.graph.add_edges_from(connect(nodes, tokens))  # edge 연결

    def get_keywords(self, limit=10, combined_keywords=False):
        scores = nx.pagerank(self.graph)  # pagerank 계산
        rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # score 역순 정렬

        _keywords = {}
        result = []
        for k in rank[:limit]:
            tuple = ()
            tuple = k[0][0],k[1],
            result.append(tuple)
            _keywords[k[0][0]] = k[1]

        keyphrases=self.get_combined_keywords(_keywords, self.tokens)
        #print('keyphrases ' + str(keyphrases))

        #we need to clear graph before a new document comes in
        self.graph.clear()

        if combined_keywords == True:
            return keyphrases

        return result

    def get_combined_keywords(self, _keywords, split_text):
        """
        :param keywords:dict of keywords:scores
        :param split_text: list of strings
        :return: combined_keywords:list
        """
        result = []
        _keywords = _keywords.copy()

        len_text = len(split_text)
        for i in range(len_text):
            word = self.strip_word(split_text[i])
            if word in _keywords:
                score = _keywords[word]
                combined_word = [word]
                if i + 1 == len_text:
                    result.append((word,score))  # appends last word if keyword and doesn't iterate
                for j in range(i + 1, len_text):
                    other_word = self.strip_word(split_text[j])
                    if other_word in _keywords and other_word == split_text[j] \
                            and other_word not in combined_word:
                        combined_word.append(other_word)
                    else:
                        for keyword in combined_word:
                            _keywords.pop(keyword)

                        result.append((" ".join(combined_word),score))
                        break

        return result

    def strip_word(self, word):
        stripped_word_list = list(self.tokenize(word))
        return stripped_word_list[0] if stripped_word_list else ""

    def tokenize(self, text, lowercase=False):
        PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)

        if lowercase:
            text = text.lower()
        for match in PAT_ALPHABETIC.finditer(text):
            yield match.group()

    def print_keywords(self, text, n=10, window_size=WINDOW_SIZE):
        print("Keyword : ")
        for k in self.keywords(text, n, window_size):
            print("{} - {}".format(k[0], k[1]))

    def summarize(self, text, max=3):
        # 자카드 유사도 계산
        def jaccard_similarity(query, document):
            intersection = set(query).intersection(set(document))
            union = set(query).union(set(document))
            return len(intersection) / len(union)

        # 문장간 유사도 측정 (BoW를 활용 코사인 유사도 측정)
        def sentence_similarity(sentence1, sentence2):
            sentence1 = [t[0] for t in self.pos_tagger.pos(sentence1) if t[1][:3] not in self.stop_pos]  # 개선 필요
            sentence2 = [t[0] for t in self.pos_tagger.pos(sentence2) if t[1][:3] not in self.stop_pos]  # 개선 필요
            # print(sentence1)
            return jaccard_similarity(sentence1, sentence2)

        def sentences(doc):
            return sent_tokenize(doc)  # [s[0].strip() for s in doc]

        def connect(doc):
            return [(start, end, sentence_similarity(start, end))
                    for start in doc for end in doc if start is not end]

        # tokens = self.pos_tagger(text)
        sentence_li = sentences(text)
        graph = nx.diamond_graph()
        graph.clear()  # 처음 생성시 graph에 garbage node가 남아있어 삭제
        graph.add_nodes_from(sentence_li)  # node 등록
        graph.add_weighted_edges_from(connect(sentence_li))  # edge 연결
        scores = nx.pagerank(graph)  # pagerank 계산

        rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # score 역순 정렬
        return rank[:max]

    def print_summarize(self, text, n=3):
        print("Summarize : ")
        for s in self.summarize(text, n):
            print("{} - {}".format(s[0], s[1]))

    def get_keywords_list(self, text, keyword_length):
        keywords = self.keywords(text, keyword_length)
        keywords = [keyword[0][0] for keyword in keywords]
        return ' '.join(keywords)

    def get_summarization_list(self, text, summarization_length):
        return ' '.join([s[0] for s in self.summarize(text, summarization_length)])

