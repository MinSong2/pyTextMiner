import os, subprocess

from sklearn.feature_extraction.text import CountVectorizer

import pyTextMiner as ptm

mecab_path='C:\\mecab\\mecab-ko-dic'
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab(mecab_path),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt'))

corpus = ptm.CorpusFromFile('./data/134963_norm.txt')
result = pipeline.processCorpus(corpus)

with open('processed_134963.txt', 'w', encoding='utf-8') as f_out:
    for doc in result:
        for sent in doc:
            new_sent = ''
            for word in sent:
                new_sent += word + ' '
            new_sent = new_sent.strip()
            f_out.write(new_sent + "\n")
f_out.close()

file_path='D:\\python_workspace\\pyTextMiner\\processed_134963.txt'
co='D:\\python_workspace\\pyTextMiner\\external_programs\\ccount.exe ' + "--input " + file_path + " --threshold " + str(2) + " --output " + "co_result.txt"

subprocess.run(co, shell=True)
co_results={}
vocabulary = {}
with open("co_result.txt", 'r', encoding='utf-8') as f_in:
    for line in f_in:
        fields = line.split()
        token1 = fields[0]
        token2 = fields[1]
        token3 = fields[2]

        tup=(str(token1),str(token2))
        co_results[tup]=float(token3)

        vocabulary[token1] = vocabulary.get(token1, 0) + 1
        vocabulary[token2] = vocabulary.get(token2, 0) + 1

        word_hist = dict(zip(vocabulary.keys(), vocabulary.values()))

graph_builder = ptm.graphml.GraphMLCreator()

#mode is either with_threshold or without_threshod
mode='with_threshold'

if mode is 'without_threshold':
    graph_builder.createGraphML(co_results, vocabulary.keys(), "test1.graphml")

elif mode is 'with_threshold':
    graph_builder.createGraphMLWithThresholdInDictionary(co_results, word_hist, "test.graphml",threshold=35.0)
    display_limit=50
    graph_builder.summarize_centrality(limit=display_limit)
    title = '동시출현 기반 그래프'
    file_name='test.png'
    graph_builder.plot_graph(title,file=file_name)
