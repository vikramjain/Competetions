import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

X = pd.read_csv('train.csv')
X.head()

null_text = X.comment_text[2]
X.shape
X.isnull().sum()
y = X[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
try:
    X.drop(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
except:
    pass
	

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_input(t):
    t = t.strip()
    print(t)	
    z = re.findall(r'[A-Za-z]+', t)
    print(z)
    z = [a for a in z if len(a) > 2]
    print(z)
    wnlemma = nltk.stem.WordNetLemmatizer()
    z = [wnlemma.lemmatize(a) for a in z]
    print(z)
    z = [a for a in z if not a in stop_words]
    print(z)
    t = ' '.join(z)
    print(t)
    return t
	
preprocess_input(null_text)

X.comment_text = X.comment_text.apply(lambda x: preprocess_input(x))

print(X.head())


vect = TfidfVectorizer(min_df=3, max_df=0.8, 
                       ngram_range=(1, 2),
                       strip_accents='unicode',
                       smooth_idf=True,
                       sublinear_tf=True,
                       )
vect = vect.fit(X['comment_text'])
X_vect = vect.transform(X['comment_text'])

print(X_vect.shape)

test = pd.read_csv('test.csv')
test.fillna(value=null_text, inplace=True)
print(test.head())

t_id = test['id']
test.drop(['id'], axis=1, inplace=True)
test.comment_text = test.comment_text.apply(lambda z: preprocess_input(z))

print(test.head())

print(len(test))

X_test = vect.transform(test['comment_text'])

print(X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_pred = pd.read_csv('sample_submission.csv')

for c in cols:
    #clf = LogisticRegression(C=4, solver='sag')
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_vect, y[c])
    y_pred[c] = clf.predict_proba(X_test)[:,1]
    pred_train = clf.predict_proba(X_vect)[:,1]
    print('log loss:', log_loss(y[c], pred_train))
	
y_pred.to_csv("my_submission.csv", index=False)