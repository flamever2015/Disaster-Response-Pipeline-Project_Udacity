import json
import plotly
from plotly.graph_objs import Bar
import os
import pandas as pd
import re
from flask import Flask
from flask import render_template, request, jsonify

# from sklearn.externals import joblib 由于有warning 所以不用该import
import joblib

import sqlalchemy
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#import plotly.graph_objs as go

app = Flask(__name__)

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

# load data
# table_name = 'DisasterResponse'
engine = create_engine('sqlite:///data/DisasterResponse.db')
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
# 务必设置路径
model = joblib.load('models/classifier.pkl')
# model = joblib.load('../models/classifier.pkl')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Top 10  categories count
    top_category_count = df.iloc[:,4:].sum().sort_values(ascending = False)[0:9]
    top_category_names = list(top_category_count.index)

    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls = plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids = ids, graphJSON = graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query = query,
        classification_result = classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug = True)


if __name__ == '__main__':
    main()
