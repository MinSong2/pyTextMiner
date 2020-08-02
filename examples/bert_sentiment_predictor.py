from textwrap import wrap

from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertForSequenceClassification

from py_bert.bert_dataset import PYBERTDataset
from py_bert.bert_classification_model import PYBERTClassifier
from py_bert.bert_predictor import bert_predictor
from py_bert.bert_trainer import PYBERTTrainer
from py_bert.bert_util import create_data_loader, add_sentiment_label, convert_to_df, get_korean_tokenizer, show_confusion_matrix
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

from py_bert.tokenization_kobert import KoBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

import pyTextMiner as ptm
import torch
import pandas as pd

#mode is either en or kr
mode = 'kr'
df = None

if mode == 'en':
    df = pd.read_csv("../data/reviews.csv")
    df, class_names = add_sentiment_label(df)
elif mode == 'kr':
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    stopwords = '../stopwords/stopwordsKor.txt'
    input_file = '../data/ratings_test.txt'

    pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.lemmatizer.SejongPOSLemmatizer(),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file=stopwords))

    corpus = ptm.CorpusFromFieldDelimitedFileForClassification(input_file, delimiter='\t', doc_index=1, class_index=2)

    documents = []
    labels = []
    result = pipeline.processCorpus(corpus)
    i = 1
    #below is just for a sample test
    for doc in result[1:500]:
        document = ''
        for sent in doc:
            for word in sent:
                document += word + ' '
        documents.append(document.strip())
        labels.append(corpus.pair_map[i])
        i += 1

    df, class_names = convert_to_df(documents,labels)

print(df.head())
print(df.info())

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

tokenizer = None
# bert-base-multilingual-cased, bert-base-cased, monologg/kobert, monologg/distilkobert, bert_models/vocab_etri.list
# bert_model_name='../bert_models/vocab_mecab.list'
bert_model_name = 'monologg/kobert'
tokenizer = get_korean_tokenizer(bert_model_name)

#we need a better way of setting MAX_LEN
MAX_LEN = 160

predictor = bert_predictor()
predictor.load_data(df, tokenizer, MAX_LEN)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#algorithm and saved_training_model goes hand-in-hand
algorithm='no_transformers'
#saved_training_model = './model_save/best_model_state.bin'
if algorithm=='transformers':
    saved_training_model = './model_save/best_model_state.bin'
else:
    saved_training_model = './model_save/best_model_states.bin'

predictor.load_model(saved_training_model)

y_texts, y_pred, y_pred_probs, y_test = predictor.predict(device, algorithm=algorithm)
print(y_pred)
print(y_test)

print(classification_report(y_test, y_pred, target_names=class_names))
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

'''
for i, (a, b) in enumerate(zip(y_test, y_pred)):
    print(classification_report(a, b, target_names=class_names))
    cm = confusion_matrix(y_test[i], y_pred[i])
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)
'''

#let’s have a look at an example from our test data:
idx = 2
text = y_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
  'class_names': class_names,
  'values': y_pred_probs[idx]
})
print("\n".join(wrap(text)))
print()
print(f'True sentiment: {class_names[true_sentiment]}')
print('\n')

#we can look at the confidence of each sentiment of our model:
sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1]);
plt.show()

text = '정말 형편없네 ㅠㅠ 눈을 버렸어'
prediction = predictor.predict_each(device,text,tokenizer,MAX_LEN, algorithm=algorithm)
print(f'Review text: {text}')
print(f'Sentiment  : {class_names[prediction]}')

#predictor.predict(device)



