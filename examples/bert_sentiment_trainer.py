from py_bert.bert_dataset import PYBERTDataset
from py_bert.bert_classification_model import PYBERTClassifier
from py_bert.bert_trainer import PYBERTTrainer
from py_bert.bert_util import create_data_loader, add_sentiment_label, convert_to_df
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

from py_bert.tokenization_kobert import KoBertTokenizer

import pyTextMiner as ptm
import torch
import numpy as np
import pandas as pd

#mode is either en or kr
mode = 'ko_distilkobert'
df = None

if mode == 'en':
    df = pd.read_csv("../data/reviews.csv")
    df, class_names = add_sentiment_label(df)
elif mode == 'kr' or mode.startswith('ko'):
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    stopwords = '../stopwords/stopwordsKor.txt'
    input_file = '../data/ratings.txt'

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
    for doc in result[1:100]:
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

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#we need a better way of setting MAX_LEN
MAX_LEN = 160

#split
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print(df_train.shape, df_val.shape, df_test.shape)

tokenizer = None
bert_model_name = ''
if mode == 'kr':
    bert_model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
elif mode == 'en':
    bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
elif mode == 'ko_bert':
    bert_model_name = 'monologg/kobert'
    tokenizer = KoBertTokenizer.from_pretrained(bert_model_name)
elif mode == 'ko_distilkobert':
    bert_model_name = 'monologg/distilkobert'
    tokenizer = KoBertTokenizer.from_pretrained(bert_model_name)

BATCH_SIZE = 16
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# print(str(train_data_loader.dataset.__getitem__(0)))
data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PYBERTClassifier(len(class_names), bert_model_name)
model = model.to(device)

num_epochs = 10
trainer = PYBERTTrainer()
trainer.train(model, device, train_data_loader, val_data_loader, df_val, df_train, num_epochs=num_epochs)


