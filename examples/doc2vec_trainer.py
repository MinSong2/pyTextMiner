if __name__ == '__main__':
    from doc2vec.doc2vecModel import Doc2VecTrainer
    import logging
    import pyTextMiner as ptm
    from gensim.models.doc2vec import TaggedDocument
    #pv_dmc, pv_dma, pv_dbow
    algorithm = 'pv_dma'
    # ignores all words with total frequency lower than this
    vocab_min_count = 10
    # word and document vector siz
    dim = 100
    # window size
    window = 5
    #number of training epochs
    epochs = 20
    # initial learning rate
    alpha = 0.025
    # learning rate will linearly drop to min_alpha as training progresses
    min_alpha = 0.001
    # number of cores to train on
    cores = 2
    # number of cores to train on
    train = True

    mecab_path = 'C:\\mecab\\mecab-ko-dic'

    # stopwords file path
    stopwords = '../stopwords/stopwordsKor.txt'
    # train documents input path
    input_path = '../data/donald.txt'
    # output base directory
    output_base_dir = './tmp'

    pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file=stopwords))

    corpus = ptm.CorpusFromFile(input_path)
    documents = []
    result = pipeline.processCorpus(corpus)
    i = 0
    for doc in result:
        document = []
        for sent in doc:
            for word in sent:
                document.append(word)
        documents.append(TaggedDocument(document, [i]))
        i += 1

    #--epochs 40 --vocab-min-count 10 data/stopwords_german.txt dewiki-preprocessed.txt /tmp/models/doc2vec-dewiki

    doc2vec = Doc2VecTrainer()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)
    doc2vec.run(documents, output_base_dir=output_base_dir, vocab_min_count=vocab_min_count,
        num_epochs=epochs, algorithm=algorithm, vector_size=dim, alpha=alpha,
        min_alpha=min_alpha, train=train, window=window, cores=cores)