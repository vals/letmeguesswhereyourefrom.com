import os
import json

from sklearn.externals import joblib
from sklearn.base import TransformerMixin 
from flask import Flask, Response, render_template

class Reducer(object):
    def __init__(self, idx):
        self.idx = idx
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
    
    def transform(self, X):
        return X[:, self.idx]
    
    def get_params(self, *k, **kw):
        return {'': ''}

name_clf = joblib.load('name_clf2.pkl')
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

#    predictions = []
#    for c, lp in zip(['q', 'w', 'e', 'r', 't'], [0.1, 0.2, 0.3, 0.4, 0.5]):
#        predictions.append({'country': c, 'probability': lp})

    return Response(json.dumps(predictions, indent=2), mimetype='application/json')


@app.route('/')
@app.route('/<name>')
def index(name='John Smith'):
    return render_template('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0', 80)
