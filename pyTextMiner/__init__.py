from pyTextMiner.splitter import *
from pyTextMiner.tokenizer import *
from pyTextMiner.stemmer import *
from pyTextMiner.lemmatizer import *
from pyTextMiner.tagger import *
from pyTextMiner.helper import *
from pyTextMiner.counter import *
from pyTextMiner.chunker import *
from pyTextMiner.cooccurrence import *
from pyTextMiner.graphml import *
from pyTextMiner.ngram import *
from pyTextMiner.segmentation import *
from pyTextMiner.keyword import *
from pyTextMiner.noun_extractor import *
from pyTextMiner.pmi import *
from pyTextMiner.collector import *

from os import listdir
import numpy as np

class Pipeline:
    def __init__(self, *pipelines):
        self.pipeline = pipelines
        self.collapse = self.checkType(pipelines)
        pass

    def checkType(self, pipeline):
        if not pipeline: return
        if pipeline[0].IN_TYPE != [str]: raise Exception("%s requires '%s' as input, but receives 'str'" % (type(pipeline[0]).__name__, pipeline[0].IN_TYPE))
        collapse = [[]]
        curType = pipeline[0].OUT_TYPE
        for p, q in zip(pipeline[:-1], pipeline[1:]):
            qt = q.IN_TYPE
            pt = p.IN_TYPE
            if qt == curType[-len(qt):]:
                collapse.append(curType[:-len(qt)])
                curType = curType[:-len(qt)] + q.OUT_TYPE
                continue
            raise Exception("%s requires '%s' as input, but receives '%s' from %s" % (type(q).__name__, qt, pt, type(p).__name__))
        return collapse

    def processCorpus(self, corpus):
        ''' process corpus through pipeline '''

        def apply(p, a, inst):
            if not a:
                return p(inst)
            if a[0] == list:
                return [apply(p, a[1:], i) for i in inst]
            if a[0] == dict:
                return {k:apply(p, a[1:], v) for k, v in inst}

        results = []
        for d in corpus:
            inst = d
            for p, c in zip(self.pipeline, self.collapse):
                inst = apply(p, c, inst)

            #a = np.array(inst)
            #print("array shape " + str(len(a.shape)) + " : " + str(isinstance(inst[0], list)))
            results.append(inst)

            # if len is 2, it is array of array
            '''
            documents = []
            for doc in inst:
                document = ''
                for sent in doc:
                    document += " ".join(sent)
                documents.append(document)
            results.append(documents)
            '''

        return results

class Corpus:
    def __init__(self, textList):
        self.pair_map = {}
        self.docs = textList

    def __iter__(self):
        return self.docs.__iter__()

    def __len__(self):
        return self.docs.__len__()

class CorpusFromFile(Corpus):
    def __init__(self, file):
        self.docs = open(file, encoding='utf-8').readlines()

class CorpusFromEojiFile(Corpus):
    def __init__(self, file):
        import re
        emoji_pattern = re.compile("["
                                   u"\U00010000-\U0010FFFF"
                                   "]+", flags=re.UNICODE)
        array = []
        line_number = 0
        with open(file, encoding='utf-8') as original:
            for line in original.readlines():
                line_number += 1
                try:
                    after = emoji_pattern.sub(r'', line)
                    array.append(after)
                except IndexError:
                    print('line number', line_number, 'txt 파일에서 확인요망')
                    array.append()
        self.docs = array


class CorpusFromFieldDelimitedFile(Corpus):
    def __init__(self, file, index):
        array = []
        line_count=0
        with open(file, encoding='utf-8') as ins:
            for line in ins.readlines():
                inside=line.split('\t')
                line_count+=1
                try:
                    in_in=inside[index]
                except IndexError:
                    print(line_count,'번째 에러 확인, txt 파일의 확인요망')
                array.append(in_in)
        self.docs = array

class CorpusFromFieldDelimitedFileWithYear(Corpus):
    def __init__(self, file, doc_index=1, year_index=0):
        array = []
        id = 0
        pair_map = {}
        with open(file, encoding='utf-8') as ins:
            for line in ins:
                fields = line.split('\t')
                try:
                    array.append(fields[doc_index])
                    pair_map[id] = fields[year_index]

                    id += 1
                except IndexError:
                    print("out of index " + str(id))

        self.docs = array
        self.pair_map = pair_map


class CorpusFromFieldDelimitedFileForClassification(Corpus):
    def __init__(self, file, delimiter='\t',doc_index=1, class_index=0, title_index=-1):
        array = []
        id = 0
        pair_map = {}
        with open(file, encoding='utf-8') as ins:
            for line in ins:
                fields = line.split(delimiter)
                try:
                    doc = ''
                    if title_index != -1:
                        doc += ' ' + fields[title_index]

                    doc += ' ' + fields[doc_index]
                    array.append(doc.strip())
                    pair_map[id] = fields[class_index]

                    id += 1
                except IndexError:
                    print("out of index " + str(id))

        self.docs = array
        self.pair_map = pair_map



class CorpusFromDirectory(Corpus):

    def __init__(self, directory, is_train):
        array = []

        # walk through all files in the folder
        for filename in listdir(directory):
            # skip any reviews in the test set
            if is_train and filename.startswith('cv9'):
                continue
            if not is_train and not filename.startswith('cv9'):
                continue

            # create the full path of the file to open
            path = directory + '/' + filename
            # load the doc

            with open(path) as myfile:
                data = "".join(line.rstrip() for line in myfile)
                #print("data :: " + data)

            # add to list
            array.append(data)

        self.docs = array


class CorpusFromFieldDelimitedEmojiFile(Corpus):
    def __init__(self, file, index):
        import re
        emoji_pattern = re.compile("["
                                   u"\U00010000-\U0010FFFF"
                                   "]+", flags=re.UNICODE)
        array = []
        line_number = 0
        with open(file, encoding='utf-8') as original:
            for line in original.readlines():
                line_split = line.split('\t')
                line_number += 1
                try:
                    text = line_split[index]
                    after = emoji_pattern.sub(r'', text)
                    array.append(after)
                except IndexError:
                    print('line number', line_number, 'txt 파일에서 확인요망')
                    array.append()
        self.docs = array
