import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier

import sklearn

class documentClassifier:
    def __init__(self):
        name='document_classifier'
        self.documents=[]
        self.features=None
        self.labels=None
        self.category_id_df=None
        self.df=None
        self.tfidf=None
        self.id_to_category=None
        self.category_to_id=None

        self.features=None
        self.labels=None
        self.class_number=0

    def preprocess(self, documents, class_list):

        #1. we assume that pyTextMiner pre-processing module was applied to the list of documents
        print(str(len(documents)) + " : " + str(len(class_list)))
        #create a list of rows consisting of text and class label by DataFrame
        self.df = pd.DataFrame({'text': documents, 'label': class_list})
        self.df = sklearn.utils.shuffle(self.df, n_samples=500, random_state=100)
        self.df.reset_index(inplace=True, drop=True)

        # Remove missing values in “text” column, and add a column encoding the label as an integer
        # because categorical variables are often better represented by integers than strings.
        # Create a couple of dictionaries for future use.
        # After cleaning up, this is the first five rows of the data we will be working on
        col = ['text', 'label']
        self.df = self.df[col]
        self.df = self.df[pd.notnull(self.df['text'])]
        self.df.columns = ['text', 'label']
        self.df['category_id'] = self.df['label'].factorize()[0]
        self.category_id_df = self.df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(self.category_id_df.values)
        self.id_to_category = dict(self.category_id_df[['category_id', 'label']].values)

        # To save the dictionary into a file:
        json.dump(self.id_to_category, open("./model/id_to_category.json", 'w'))

        print(self.df.head())

        #number of classes
        self.class_number=self.df.label.unique()

        #visualize the distribution of classes
        fig = plt.figure(figsize=(8,6))
        self.df.groupby('label').text.count().plot.bar(ylim=0)
        plt.show()

        #2. vectorization
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8')
        self.features = self.tfidf.fit_transform(self.df.text).toarray()
        self.labels = self.df.category_id
        print(self.features.shape)

        #3. feature selection
        # We can use sklearn.feature_selection.chi2 to find the terms
        # that are the most correlated with each of the class labels
        N = 2
        for label, category_id in sorted(self.category_to_id.items()):
            features_chi2 = chi2(self.features, self.labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(self.tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(label))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    def train(self, model_index=1):
        #4. supervised-based classification model selection
        # We are now ready to experiment with different machine learning models,
        # evaluate their accuracy and find the source of any potential issues.
        # We will benchmark the following four models:
        # Logistic Regression
        # (Multinomial) Naive Bayes
        # Linear Support Vector Machine
        # Random Forest
        models = [
            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=0),
            KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),
            SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001)
        ]

        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, self.features, self.labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()

        cv_df.groupby('model_name').accuracy.mean()

        #5. data preparation for training and validating
        # Continue with our best model (LinearSVC), we are going to look at the confusion matrix,
        # and show the discrepancies between predicted and actual labels.
        model = models[model_index]

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(self.features, self.labels, self.df.index,
                                                                                         test_size=0.33, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return X_train, X_test, y_train, y_test, y_pred, indices_test, model

    def evaluate(self, y_test, y_pred, indices_test, model):
        #6. evaluation
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=self.category_id_df.label.values, yticklabels=self.category_id_df.label.values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        # The vast majority of the predictions end up on the diagonal (predicted label = actual label),
        # where we want them to be. However, there are a number of misclassifications,
        # and it might be interesting to see what those are caused by:
        for predicted in self.category_id_df.category_id:
            for actual in self.category_id_df.category_id:
                if predicted != actual and conf_mat[actual, predicted] >= 10:
                    print("'{}' predicted as '{}' : {} examples.".format(self.id_to_category[actual], self.id_to_category[predicted],
                                                                         conf_mat[actual, predicted]))
                    print(self.df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][
                                ['label', 'text']])
                    print('')

        # As you can see, some of the misclassified complaints are complaints that touch on more than one subjects
        # (for example, complaints involving both credit card and credit report). This sort of errors will always happen.
        # Again, we use the chi-squared test to find the terms that are the most correlated with each of the categories:
        model.fit(self.features, self.labels)
        N = 2
        if 'kneighbors' not in model.__class__.__name__.lower():
            for label, category_id in sorted(self.category_to_id.items()):
                indices = np.argsort(model.coef_[category_id])
                feature_names = np.array(self.tfidf.get_feature_names())[indices]
                unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
                bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
                print("# '{}':".format(label))
                print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
                print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))



        #8. final evaluation per class
        print(metrics.classification_report(y_test, y_pred,
                                            target_names=self.df['label'].unique()))

    def save(self, model, model_name='classification.model'):
        dump(model, model_name)

    def saveVectorizer(self, model_name='vectorizer.model'):
        dump(self.tfidf, model_name)

    def load(self, model_name):
        return load(model_name)

    def loadVectorizer(self, model_name='vectorizer.model'):
        return load(model_name)

    def predict(self, model, vectorizer_model):
        #7. prediction
        docs = ["한국 경제 글로벌 위기 수요 위축 시장 경제 붕귀 자동차 수출 빨간불 내수 촉진 증진 방향성 제고",
                "밝기 5등급 정도 도심 밖 맨눈 충분히 관측 가능 새해 미국인 8월 행운 기대",
                "최순실 민간인 국정농단 의혹 사건 진상규명 국정조사 특별위원회 1차 청문회 이재용 삼성전자 부회장 재벌 총수 9명 증인 출석"]

        with open("./model/id_to_category.json") as handle:
            id_to_category = json.loads(handle.read())

        text_features = vectorizer_model.transform(docs)
        predictions = model.predict(text_features)
        for text, predicted in zip(docs, predictions):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(id_to_category[str(predicted)]))
            print("")

    def predict_realtime(self, model, vectorizer_model, docs):

        with open("./model/id_to_category.json") as handle:
            id_to_category = json.loads(handle.read())

        text_features = vectorizer_model.transform(docs)
        predictions = model.predict(text_features)
        for text, predicted in zip(docs, predictions):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(id_to_category[str(predicted)]))
            print("")