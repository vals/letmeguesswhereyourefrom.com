# Let me guess where you're from

On the website [letmeguesswhereyourefrom.com](http://www.letmeguesswhereyourefrom.com/) you can enter a name, and an algorithm will print five countries the name seems come from. Try it out!

[FIGURE]

I made this web application as an exercise in applying machine learning, in this post I will describe 

 - How I got data,
 - How I made the model,
 - How I optimized the model for use in a web application, and,
 - How I made the web application.

### Getting training data

The first problem is where to get the data from. For the model to be representative I would want data from real people. There are many lists of "Most common names in ..." for various countries. These however mostly consist of either first names or surnames. I doubted just for example first names would not contain enough information to train a model. For a while I thought I could get surname lists and first name lists and randomly pair elements from them to make larger training sets. Looking for real names, I found that Wikipedia has [a list of people who have Wikipedia pages by country](https://en.wikipedia.org/wiki/Lists_of_people_by_nationality)!

We can quickly get all this data

    %pylab inline
    import pandas as pd
    import seaborn as sns
    sns.set_color_codes()
    
    import codecs
    from unidecode import unidecode
    
    import requests
    
    wiki = 'https://en.wikipedia.org'
    res = requests.get(wiki + '/wiki/Lists_of_people_by_nationality')
    
<br>
By inspecting the downloaded HTML I noticed a pattern on the rows that contain links to lists of people. So I parsed these and extract the country name and URL to the list.

    list_urls = {}
    for line in res.content.split('\n'):
        if line.startswith('<li><a href="/wiki/List_of_'):
            l =  line.split('<li><a href="')[-1] \
                     .split('" title=')[0]
            country = l.split('/wiki/List_of_')[-1]
            list_urls[country] = l

    f1 = lambda s: s.startswith('<li><a href="/wiki/')
    c1 = lambda s: s.split('" title="')[-1].split('"')[0]

    lists = {}
    for country in list_urls:
        content = requests.get(wiki + list_urls[country]).text.split('\n')
        lists[country] = map(unidecode, map(c1, filter(f1, content)))

    with open('name_list.tsv', 'w') as fh:
	    for country in lists:
	        for name in lists[country]:
	            fh.write('{}\t{}\n'.format(name, country))

<br>
There are some obvious problems with this. Not all lines I parse out are actually lists of names, but other lists. Additionally the larger countries have so many entries that they link to lists of sublists rather than actually lists. (This includes US and China.) The number of countries isn't that high and it would be possible to just add these manually. But I don't feel like doing that and will ignore those cases.

To make downstream analysis easier, I romanized all names using the unidecode package. This translates all characters in the names to ASCII.

In total the process fetches in total 68,634 names distributed over 214 countries.

### Making the model

I knew that N-grams of words are commonly used for document classification, and thought I can make something similar for just letters in names. Turns out this was already nicely implemented in the scikit learn as a parameter to the CountVectorizer class.

For the most part I followed the [scikit-learn text analytics tutorial](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). The tutorial said "bags of words" representations of documents typically have more than 100 0000 features. Since a "document" in my case is a name, there are not so many n-grams in each. To get a large number of featuers I set the CountVectorizer to consider all N-grams for all N between 1 and 5. I limited the maximum number of features to 100000 so that I would know how large the data sets would be.

As per the tutorial, I push the output of the CountVectorizer through a TfidfTransformer. The idea with this is to downscale weights of features that occur in many samples.

Finally, I use these features to train a [Multinomial Naive Bayes model](http://scikit-learn.org/stable/modules/naive_bayes.html). (I'm not discussing much of the details of the methods here, as they are well described in the references. A thing about scikit-learn for applying machine learning that I like is also that you can use a lot of very nice tools without knowing much about the details.)

All these steps are combined to a pipeline, so we can easily just supply raw text and get a categorical prediction.

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import train_test_split

    name_clf = Pipeline([('vect', CountVectorizer(analyzer='char',
                                              	  ngram_range=(1,5),
                                              	  max_features=100000)),
                     	('tfidf', TfidfTransformer()),
                     	('clf', MultinomialNB())])

<br>
The data is read in, and split in to a training set that we will use to tune the model, and a testing set that we will finally use to evaluate the model.

    df = pd.read_table('name_list.tsv', header=-1)
    df.columns = ['name', 'country']
    name_train, name_test, country_train, country_test = \
    train_test_split(df.name, df.country, test_size=10000)

<br>
Now I could simply train the classifier using the training data.

    name_clf.fit(name_train, country_train)

    predicted = name_clf.predict(name_test)
	np.mean(predicted == country_test)

	Out []: 0.27860000000000001

<br>
While much better than randomly guessing countries (1 / 214 = 0.0047), it's not a particularly impressive score.

Earlier the same day, I had listened to the [Talking Machines](http://www.thetalkingmachines.com/) podcast where [Jennifer Listgarten](http://research.microsoft.com/en-us/um/people/jennl/) described how they had trained a model to predict efficient CRISPR/Cas9 guide targets (avilable in a soon to be released service [Azimuth](http://research.microsoft.com/en-us/projects/azimuth)). She said that predicting _the best_ guide was a very hard problem. But predicting say the top 5 guides had very high accuracy. Trying a small number of guides is much more efficient than many possible guides anyway.

So I stole that strategy, and return a top 5 of predicted countries for a given name. Thus I change the model evaluation criterion. The strategy is very well suited for MultinomialNB, since prediction means evaluating the posterior probability for each class. We can just take these probabilities and sort out the highest ones.

    t5 = name_clf.predict_log_proba(name_test).argsort(axis=1)[:,-5:]
    a = country_test
    b = name_clf.get_params()['clf'].classes_[t5]
    np.mean((a[:, None] == b).max(1))

    Out []: 0.4647

<br>
Better, but still not stellar.

The MultinomialNB model has a hyperparameter alpha. This is a smoothing parameter that helps with the problem that many features have 0 counts in any given document. This would cause the posterior probability to be 0. The default value for alpha is 1. We'll try to do better by optimizing the alpha by randomized cross validation.

    from sklearn.grid_search import RandomizedSearchCV

    a = 10 ** np.linspace(-5, 0)
	rscv = RandomizedSearchCV(name_clf,
    param_distributions={'clf__alpha': a},
    n_iter=10,
    n_jobs=2)

    rscv.fit(name_train, country_train)

    rscv.best_params_

    Out []: {'clf__alpha': 0.014563484775012445}

<br>
We plug this value in to out classifier pipeline, and try again.

    name_clf = Pipeline([('vect', CountVectorizer(analyzer='char',
                                                  ngram_range=(1,5),
                                                  max_features=100000)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB(alpha=0.014563484775012445))])
	
	name_clf.fit(name_train, country_train)

	t5 = name_clf.predict_log_proba(name_test).argsort(axis=1)[:,-5:]
	a = country_test
	b = name_clf.get_params()['clf'].classes_[t5]
	np.mean((a[:, None] == b).max(1))

	Out []: 0.66959999999999997

<br>
This is quite nice! We can make a function to check how it performs in practice.

    def top_countries(name):
        log_proba = name_clf.predict_log_proba([name])
        t5 = log_proba.argsort(axis=1)[:,-5:]
        b = name_clf.get_params()['clf'].classes_[t5]
        for c, lp in zip(b[0], log_proba[:,t5].flatten())[::-1]:
            print '{}\t\t{}'.format(c, np.exp(lp))

	top_countries('angela merkel')

	Vietnamese		0.893538897127
	Greeks		0.0294758574644
	Baltic_Germans		0.0194029861994
	Irish_people		0.015581222622
	Bulgarians		0.00674997108663

<br>
While some of the suggestions are... counterintuitive... there is some German there! The strategy to return a top 5 seem to work out.

### Making the model deployable

While I thought it would be nice to make the model usable through a web application, I didn't want to spend a fortune doing so! [Heroku](https://www.heroku.com/pricing) offers free web application hosting in their Free tier with the limitation of 512 MB of memory. Additionally, the final size of the compiled web application "slug" can only be 300 MB.

The training data is 58 000 examples with 100 000 features each. The MultinomialNB model needs to store two dense matrices of size (58 000, 100 000). Running the model in the Python interpreter requires a bit over 1 GB of RAM. Storing the model with joblib.dump() as recommended in the [documentation](http://scikit-learn.org/stable/modules/model_persistence.html) creates around 700 MB of data, so we wouldn't be able to deploy it on Heroku.

The strategy I settled for was to put in a step in the pipeline to ignore non-informative features. To find which features contain the most information I used TruncatedSVD, a dimensionality reduction technique known to work well for "bags of words" models.

    from sklearn.decomposition import TruncatedSVD

    tf_pipeline = Pipeline([('vect', CountVectorizer(analyzer='char',
       		                                         ngram_range=(1,5),
              		                                 max_features=100000)),
	                        ('tfidf', TfidfTransformer())])
	
	name_features = tf_pipeline.fit_transform(name_train)

	tsvd = TruncatedSVD(n_components=400)

	tsvd.fit(name_features)

	tsvd.explained_variance_ratio_.sum()

	Out []: 0.27138872714390422

<br>
I picked 400 components because sqrt(100000) = 316.23 and I rounded up. There's no rule for this or so, but this let me fit the TruncatedSVD for the entire training set in a reasonable amount of time. The total variance explained by 400 components is about 27%. It doesn't sound as much, but it can inform us on the importance of the features.

We extract importances in the TruncatedSVD by looking at the lengths of the singular vectors stored in TruncatedSVD.components_.

    idx = np.linalg.norm(tsvd.components_[:, :], axis=0).argsort()

We make a "Reducer" class which we can plug in to the pipeline to tell the pipeline which features to pass on from the feature extraction step to the MultinomialNB classifier.

    from sklearn.base import TransformerMixin

    class Reducer(TransformerMixin):
    def __init__(self, idx):
        self.idx = idx
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
    
    def transform(self, X):
        return X[:, self.idx]
    
    def get_params(self, *k, **kw):
        return {'': ''}

<br>
We could include the important feature selection in a fit method, but to speed this up, we perform that calculation offline and provide the indexes of the features to the constructor of the object.

I checked a range of the number of features to pass on to the MultinomialNB by looking at the model score. This is bad thing to do and you shouldn't do it! Now I'm causing an information leak between the parameter selection and the model evaluation. However, I think it's relatively obvious that the score will be monotonically increasing with the number of features. And I had already decided I wanted to use between 10 000 and 20 000 features for the final model due to the computational budget. I was just curious at what rate things _would_ get better _if_ I'd allow more features.

[FIGURE]

Now we train the model with a reducer step that passes the 15 000 most informative features to the classifier step.

    name_clf = Pipeline([('vect', CountVectorizer(analyzer='char',
                                                  ngram_range=(1,5),
                                                  max_features=100000)),
                         ('tfidf', TfidfTransformer()),
                         ('reducer', Reducer(idx[-15000:])),
                         ('clf', MultinomialNB(alpha=0.014563484775012445))])
	
	name_clf.fit(name_train, country_train)

	t5 = name_clf.predict_log_proba(name_test).argsort(axis=1)[:,-5:]
	a = country_test
	b = name_clf.get_params()['clf'].classes_[t5]
	s = np.mean((a[:, None] == b).max(1))
	s

	Out []: 0.62849999999999995

<br>
A pretty modest performance decrease given that the model only takes 15% of the memory needed before! As a side effect, the prediction operation in the model is now a lot snappier as well, which feels good in the web app.

With the indices for the features saved, it is actually relatively quick to just train the model from the total amount of raw data. So for deploying the web app, I just have it load the data and train when it launches. It takes about 30 seconds, and the raw data (the list of names and countries) is about 2 MB in total.

### Web application

The web application itself is written using the microframework Flask. I define one entry point that provides an API, where the name is queried and the predictor is called to return the top scoring countries. The API returns these in JSON format. The second entry point just serves a HTML file, which in turn loads a Javascript file that connects the input box and buttons to the API.

To get access to scientific Python tools like scikit-learn on Heroku I use the [Conda Buildpack](https://github.com/kennethreitz/conda-buildpack).

You can see the full implementation of the app on [GitHub](https://github.com/vals/letmeguesswhereyourefrom.com), as well as the [Jyputer notebook](https://github.com/vals/letmeguesswhereyourefrom.com/blob/master/Making%20the%20name%20classifier.ipynb) where I made the model.

Finally, have a look at the result!

[letmeguesswhereyourefrom.com](http://www.letmeguesswhereyourefrom.com/)