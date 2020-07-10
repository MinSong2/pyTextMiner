from topic_model.MalletWrapper import MalletTopicModel

model = MalletTopicModel('D:\python_workspace\pyTextMiner\mallet')
#model.import_file(input=r'C:\mallet\topic_input\dblp_sample.txt')
model.import_file(input=r'D:\python_workspace\pyTextMiner\mallet\topic_input\sample_dmr_input.txt')
model.train_topics()

#print(model.topic_keys)  # see output_topic_keys parameter in Train Topics documentation
# print(model.doc_topics)  # see output_doc_topics parameter in Train Topics documentation
#print(model.word_weights)  # see topic_word_weights_file parameter in Train Topics documentationn