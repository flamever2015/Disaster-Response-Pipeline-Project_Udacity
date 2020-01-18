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
from sklearn.externals import joblib

import random
import pickle
import warnings

def load_data(database_filepath):
    # 用 sqlalchemy 构建数据库链接engine
    # table name
    table_name = 'DisasterResponse'
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    
    # df = pd.read_csv(database_filepath, low_memory = False)
    X = df.message
    Y = df.iloc[:, 4:]
    #Y = df.iloc[:, 4]
    #category_names = pd.DataFrame(Y).columns.value
    category_names = pd.DataFrame(Y).columns
    return X, Y, category_names


def tokenize(text):
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
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))])
        #('clf', RandomForestClassifier())])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 100],
        'clf__estimator__min_samples_split': [2, 4]

    }

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # get predictions 
    y_preds = model.predict(X_test)
    # print classification report
    print(classification_report(y_preds, Y_test.values, target_names = category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_preds)))


def save_model(model, model_filepath):
    # joblib.dump(model, model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))


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
