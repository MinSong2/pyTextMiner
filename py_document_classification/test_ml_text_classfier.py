from document_classification.ml_textclassification import documentClassifier
import pyTextMiner as ptm

if __name__ == '__main__':
    document_classifier = documentClassifier()
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            #ptm.tokenizer.LTokenizerKorean(),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )

    #mode is either train or predict
    mode = 'train'
    if mode is 'train':
        input_file ='./data/3_class_naver_news.csv'
        # 1. text processing and representation
        corpus = ptm.CorpusFromFieldDelimitedFileForClassification(input_file,
                                                                   delimiter=',',
                                                                   doc_index=4,
                                                                   class_index=1,
                                                                   title_index=3)
        corpus.docs
        tups = corpus.pair_map
        class_list = []
        for id in tups:
            #print(tups[id])
            class_list.append(tups[id])

        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        document_classifier.preprocess(documents,class_list)

        #model_name = 0  -- RandomForestClassifier
        #model_name = 1  -- LinearSVC
        #model_name = 2  -- MultinomialNB
        #model_name = 3  -- LogisticRegression
        #model_name = 4  -- K-NN
        #model_name = 5  -- SGDClassifier
        X_train, X_test, y_train, y_test, y_pred, indices_test, model = document_classifier.train(model_index=1)

        print('training is finished')

        document_classifier.evaluate(y_test,y_pred,indices_test,model)
        document_classifier.save(model, model_name='./model/svm_classifier.model')
        document_classifier.saveVectorizer(model_name='./model/vectorizer.model')

    elif mode is 'predict':
        model=document_classifier.load('./model/svm_classifier.model')
        vectorizer_model=document_classifier.loadVectorizer(model_name='./model/vectorizer.model')
        document_classifier.predict(model,vectorizer_model)

        #7. prediction
        input = "../data/navernews.txt"
        corpus = ptm.CorpusFromFieldDelimitedFile(input,3)

        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        document_classifier.predict_realtime(model,vectorizer_model, documents)
