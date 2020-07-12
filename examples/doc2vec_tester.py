if __name__ == '__main__':
    from doc2vec.doc2vecModel import Doc2VecSimilarity
    import logging
    import pyTextMiner as ptm

    model_file = '../doc2vec/tmp/1594484106304_pv_dma_dim=100_window=5_epochs=20/doc2vec.model'
    doc2vec = Doc2VecSimilarity()
    doc2vec.load_model(model_file)

    test_sample = '한국 경제가 위기에 처하다'
    # Convert the sample document into a list and use the infer_vector method to get a vector representation for it
    new_doc_words = test_sample.split()
    similars = doc2vec.most_similar(test_sample)
    for sim in similars:
        print(str(sim))

    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    # stopwords file path
    stopwords = '../stopwords/stopwordsKor.txt'

    test_sample1 = '중국 시장은 위축되었다'

    pipeline = ptm.Pipeline(ptm.tokenizer.MeCab(mecab_path),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file=stopwords))

    doc_vec1 = pipeline.processCorpus([test_sample])
    doc_vec2 = pipeline.processCorpus([test_sample1])

    print(doc_vec1[0])
    print(doc_vec2[0])

    # use the most_similar utility to find the most similar documents.
    similarity = doc2vec.compute_similarity_vec(first_vec=doc_vec1[0], second_vec=doc_vec2[0])
    print('similarity between two document: ')
    print(str(similarity))


