# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])
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
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import sys
import pickle 
from sklearn.externals import joblib

def load_data(database_filepath):
    # 用 sqlalchemy 构建数据库链接engine
    #connect_info = 'sqlite:////data/DisasterResponse.db'
    #engine = create_engine(connect_info)
    # sql 命令
    #sql_cmd = 'SELECT * FROM DisasterResponse'
    #df = pd.read_sql(sql = sql_cmd, con = engine)

    #conn = sqlite3.connect('data/DisasterResponse.db')

    # get a cursor
    #cur = conn.cursor()

    # create the test table including project_id as a primary key
    #df = pd.read_sql("SELECT * FROM DisasterResponse", con = conn)

    #conn.commit()
    #conn.close()

    #不用 engine = create_engine('sqlite:////home/workspace/df_clean.db')
    #不用 df = pd.read_sql("SELECT * FROM df_clean", engine)
    
    df = pd.read_csv(database_filepath)
    X = df.message
    #Y = df.iloc[:, 4:]
    Y = df.iloc[:, 4]
    category_names = pd.DataFrame(Y).columns.values
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf', MultiOutputClassifier(RandomForestClassifier()))])
        ('clf', RandomForestClassifier())])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        #clf__n_estimators': [50, 100],
        'clf__min_samples_split': [2, 3, 4]

    }

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    #print(r'Y_pred.shape:',Y_pred.shape)

    print(category_names, 'report:', classification_report(Y_test, Y_pred))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


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