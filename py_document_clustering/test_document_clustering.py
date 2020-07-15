
import pyTextMiner as ptm
from document_clustering.documentclustering import DocumentClustering

if __name__ == '__main__':
    corpus = ptm.CorpusFromFieldDelimitedFile('../data/donald.txt', 2)

    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            #ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )
    result = pipeline.processCorpus(corpus)
    print('==  ==')

    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            document += " ".join(sent)
        documents.append(document)

    print(len(documents))
    #name either k-means, agglo, spectral_cocluster
    name = 'spectral_cocluster'
    clustering=DocumentClustering(k=5)
    #n_components means the number of words to be used as features
    clustering.make_matrix(documents,n_components=-1)
    clustering.cluster(name)
    clustering.print_results()
