
from torch.utils.data import Dataset, DataLoader
from py_bert.bert_dataset import PYBERTDataset
import pandas as pd

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
