# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlalchemy
from sqlalchemy import create_engine
import sqlite3
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import sys
import pickle
#from sklearn.externals import joblib
import joblib
import random
import pickle
import warnings

def load_data(database_filepath):
    '''
     从数据库文件加载数据，得到dataframe
     输入：
         database_filepath：sql数据库的文件路径
     输出：
         X：消息数据（feature）
         Y：类别数据（target）
         category_names：类别名称
    '''
    
    # 用 sqlalchemy 构建数据库链接engine
    # table name
    table_name = 'DisasterResponse'
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)

    # df = pd.read_csv(database_filepath, low_memory = False)
    X = df.message
    Y = df.iloc[:, 4:]
    # "related" column has 0, 1, 2 values
    # change 2 to 1, so that the value is only 1 or 0
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)

    category_names = pd.DataFrame(Y).columns

    return X, Y, category_names


def tokenize(text):
    '''
     标记和清理文本
     输入：
         文本：原始消息文本
     输出：
         文本：经过标记化，清除和词形化的文本
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # stopword list
    STOPWORDS = list(set(stopwords.words('english')))

    clean_tokens = []
    for tok in tokens:
        if (tok.lower() not in STOPWORDS):
            # put words into base form
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", clean_tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    '''
     使用TFIDF，RandomForest分类器和gridsearch建立机器学习管道
     输入：None
     输出：GridSearchCV的结果
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
        ])
        #('clf', RandomForestClassifier())])

    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 25],
        'clf__estimator__min_samples_split': [2, 5]

    }

    cv = GridSearchCV(pipeline, param_grid = parameters, cv=3, verbose=10)
    #cv = GridSearchCV(pipeline, param_grid = parameters, cv=3, verbose=10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    '''
     使用测试数据来评估模型
     输入：
         模型：要评估的模型
         X_test：测试数据
         Y_test：测试数据的真实标签
         category_names：类别标签
     输出：
         每个类别的准确性、分类报告
    '''

    Y_pred = model.predict(X_test)
    
    # print scores
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))
    # print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))


def save_model(model, model_filepath):

    '''
     将模型保存为pickle文件
     输入：
         模型：需要保存的模型
         model_filepath：保存路径
     输出：
         已保存模型的pickle文件
    '''

    joblib.dump(model, model_filepath)
    # pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #X, Y, category_names = load_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
