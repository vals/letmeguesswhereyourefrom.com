import os
import json

import numpy as np

from sklearn.externals import joblib
from sklearn.base import TransformerMixin 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from flask import Flask, Response, render_template

name_data = np.genfromtxt('name_list.tsv', delimiter='\t', dtype=str)

names = name_data[:, 0]
countries = name_data[:, 1]

idx = np.genfromtxt('idx.csv', dtype=int)

class Reducer(object):
    def __init__(self, idx):
        self.idx = idx
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
    
    def transform(self, X):
        return X[:, self.idx]
    
    def get_params(self, *k, **kw):
        return {'': ''}

name_clf = Pipeline([('vect', CountVectorizer(analyzer='char',
                                              ngram_range=(1,5),
                                              max_features=100000)),
                     ('tfidf', TfidfTransformer()),
                     ('reducer', Reducer(idx)),
                     ('clf', MultinomialNB(alpha=0.014563484775012445))])

name_clf.fit(names, countries)

class_names = name_clf.get_params()['clf'].classes_

app = Flask(__name__)


@app.route('/guess/<name>')
def guess(name='test'):
    print name
    log_proba = name_clf.predict_log_proba([name])
    t5 = log_proba.argsort(axis=1)[:,-5:]
    b = class_names[t5]
    predictions = []
    for c, lp in zip(b[0], log_proba[:, t5].flatten())[::-1]:
        predictions.append({'country': c, 'probability': lp})

    return Response(json.dumps(predictions, indent=2), mimetype='application/json')


@app.route('/')
@app.route('/<name>')
def index(name='John Smith'):
    return render_template('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0', 8000)
