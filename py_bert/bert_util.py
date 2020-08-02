
from torch.utils.data import Dataset, DataLoader
from py_bert.bert_dataset import PYBERTDataset
import pandas as pd
from transformers import BertModel, BertTokenizer
from py_bert.tokenization_kobert import KoBertTokenizer
from py_bert.tokenization_korbert import KorBertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_korean_tokenizer(bert_model_name):
    tokenizer = None
    if bert_model_name.startswith('monologg'):
        tokenizer = KoBertTokenizer.from_pretrained(bert_model_name)
    elif 'etri' or 'mecab' in bert_model_name:
        tokenizer = KorBertTokenizer.from_pretrained(os.path.abspath(bert_model_name))
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    return tokenizer

def to_sentiment(rating):
    '''
        assuming the class rating scale is from 0 to 5
    '''
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

def add_sentiment_label(df):
    df['sentiment'] = df.score.apply(to_sentiment)
    if len(df['sentiment'].unique()) == 2:
        class_names = ['positive', 'negative']
    elif len(df['sentiment'].unique()) == 3:
        class_names = ['positive', 'neutral', 'negative']

    return df, class_names

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PYBERTDataset(
        contents=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

def convert_to_df(documents, labels):
    pd.set_option('display.max_columns', None)
    document_df = pd.DataFrame()
    combined = zip(documents,labels)
    for i, (text, label) in enumerate(combined):
        document_df = document_df.append(pd.Series([text, int(label)]), ignore_index=True)

    document_df.columns = ['content', 'sentiment']
    class_names = []
    if len(document_df['sentiment'].unique()) == 2:
        class_names = ['positive', 'negative']
    elif len(document_df['sentiment'].unique()) == 3:
        class_names = ['positive', 'neutral', 'negative']

    return document_df, class_names


def convert_to_df_for_classification(documents, labels):
    pd.set_option('display.max_columns', None)
    document_df = pd.DataFrame()
    combined = zip(documents,labels)
    for i, (text, label) in enumerate(combined):
        document_df = document_df.append(pd.Series([text, int(label)]), ignore_index=True)

    document_df.columns = ['content', 'label']
    class_names = document_df['label'].unique()

    return document_df, class_names

def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.show()

def token_count_distribution(df, tokenizer):
    token_lens = []
    for txt in df.content:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))

    sns.distplot(token_lens)
    plt.xlim([0, 256])
    plt.xlabel('Token count')

    plt.show()