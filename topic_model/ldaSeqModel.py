import numpy  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel, ldamodel
from gensim.corpora import Dictionary
import os.path
import logging

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder

class ldaSeqModel():

    def __init__(self):
        name = 'ldaSeqModel'

    def run(self, document_collection, topic_count=2, time_group=[10,10,11]):
        """document_collection should be sorted in order of time_slice."""
        dictionary = Dictionary(document_collection)
        corpus = [dictionary.doc2bow(text) for text in document_collection]
        ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, num_topics=topic_count, time_slice=time_group)

        topics = ldaseq.print_topics(1)
        for topic in topics:
            print("TOPIC " + str(topic))

        return ldaseq

    def parseDocuments(self, document_file, year_index, document_index):
        """make document along with time information"""
        dict = {}
        with open(document_file, encoding='utf-8') as ins:
            for line in ins:
                #print("LINE " + line)
                fields = line.split('\t')
                _year = fields[year_index]
                _document = fields[document_index]

                if _year not in dict:
                    d = []
                    d.append(_document)
                    dict[_year] = d

                else:
                    print("DOC " + _year)
                    _docu_ = dict.get(_year)
                    _docu_.append(_document)
        return dict

    def parseProcessedText(self, processed_documents, pair_map):
        """make document along with time information"""
        dict = {}
        for doc in processed_documents:
            for line in doc:
                #print("LINE " + line)
                fields = line.split('\t')
                _year = fields[year_index]
                _document = fields[document_index]

                if _year not in dict:
                    d = []
                    d.append(_document)
                    dict[_year] = d

                else:
                    print("DOC " + _year)
                    _docu_ = dict.get(_year)
                    _docu_.append(_document)
        return dict

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    document_file = "../time_test.txt"
    year_index = 0
    document_index = 1
    _dict = ldaSeqModel().parseDocuments(document_file, year_index, document_index)

    #import pyTextMiner as ptm
    #corpus = ptm.CorpusFromFieldDelimitedFileWithYear('time_test.txt', 1, 0)
    #pair_map = corpus.pair_map

    time_slice = []
    key_size = len(_dict)
    doc_coll = _dict.values()
    for k, v in _dict.items():
        time_slice.append(len(v))
    ldaSeqModel().run(doc_coll,5,time_slice)
