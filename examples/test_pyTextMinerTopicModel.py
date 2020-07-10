from topic_model.pyTextMinerTopicModel import pyTextMinerTopicModel
import pyTextMiner as ptm

if __name__ == '__main__':

    mecab_path='C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='./stopwords/stopwordsKor.txt')
                            )

    corpus = ptm.CorpusFromFieldDelimitedFileWithYear('./mallet/topic_input/sample_dmr_input.txt',doc_index=2,year_index=1)
    pair_map = corpus.pair_map

    result = pipeline.processCorpus(corpus.docs)
    text_data = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        text_data.append(new_doc)

    topic_model = pyTextMinerTopicModel()
    topic_number=10
    mdl=None
    #mode is either lda, dmr, hdp, infer, etc
    mode='infer'
    label=''
    if mode is 'lda':
        print('Running LDA')
        label='LDA'
        lda_model_name = './test.lda.bin'
        mdl=topic_model.lda_model(text_data, lda_model_name, topic_number)

        print('perplexity score ' + str(mdl.perplexity))

    elif mode is 'dmr':
        print('Running DMR')
        label='DMR'
        dmr_model_name='./test.dmr.bin'
        mdl=topic_model.dmr_model(text_data, pair_map, dmr_model_name, topic_number)

        print('perplexity score ' + str(mdl.perplexity))

    elif mode is 'hdp':
        print('Running HDP')
        label='HDP'
        hdp_model_name='./test.hdp.bin'
        mdl, topic_num=topic_model.hdp_model(text_data, hdp_model_name)
        topic_number=topic_num
    elif mode is 'hlda':
        print('Running HLDA')
        label='HLDA'
        hlda_model_name = './test.hlda.bin'
        mdl=topic_model.hlda_model(text_data, hlda_model_name)
    elif mode is 'infer':
        lda_model_name = './test.lda.bin'
        unseen_text='아사이 베리 블루베리 비슷하다'
        topic_model.inferLDATopicModel(lda_model_name, unseen_text)

    if (mode is not 'infer'):
        # The below code extracts this dominant topic for each sentence
        # and shows the weight of the topic and the keywords in a nicely formatted output.
        df_topic_sents_keywords, matrix = topic_model.format_topics_sentences(topic_number=topic_number, mdl=mdl)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        df_dominant_topic.head(10)

        # Sometimes we want to get samples of sentences that most represent a given topic.
        # This code gets the most exemplar sentence for each topic.
        topic_model.distribution_document_word_count(df_topic_sents_keywords, df_dominant_topic)

        #When working with a large number of documents,
        # we want to know how big the documents are as a whole and by topic.
        #Let’s plot the document word counts distribution.
        topic_model.distribution_word_count_by_dominant_topic(df_dominant_topic)

        # Though we’ve already seen what are the topic keywords in each topic,
        # a word cloud with the size of the words proportional to the weight is a pleasant sight.
        # The coloring of the topics I’ve taken here is followed in the subsequent plots as well.
        ##topic_model.word_cloud_by_topic(mdl)

        # Let’s plot the word counts and the weights of each keyword in the same chart.
        topic_model.word_count_by_keywords(mdl,matrix)

        # Each word in the document is representative of one of the N topics.
        # Let’s color each word in the given documents by the topic id it is attributed to.
        # The color of the enclosing rectangle is the topic assigned to the document.
        topic_model.sentences_chart(mdl,start=0, end=5, topic_number=topic_number)

        #visualize documents by tSNE
        topic_model.tSNE(mdl,matrix,label,topic_number=10)

        topic_model.make_pyLDAVis(mdl,matrix,text_data)
