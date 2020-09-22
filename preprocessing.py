import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import string

import os
from scikit_roughsets.rs_reduction import RoughSetsSelector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='ar', timeout=30000)

def normalize(word):
    word = str(word)
    word = word.replace('أ','ا')
    word = word.replace('آ','ا')
    word = word.replace('إ','ا')
    word = word.replace('ـ','')
    word = word.replace('ة','ه')
    word = word.replace('ٌ','')
    word = word.replace('ْ','')
    word = word.replace('ٍ','')
    word = word.replace('ِ','')
    word = word.replace('ٌ','')
    word = word.replace('ً','')
    word = word.replace('َ','')
    if len(word) == 1:
        word = ''
    return word

df = pd.read_excel('Complaints3.xlsx',sheet_name='Sheet1')
preprocessed = pd.DataFrame(columns=['Title','Category'])
df_stopwords = pd.read_csv('arabicStopWords')
stopwords = []
for i in df_stopwords:
    stopwords.append(normalize(str(i[0])))
print(stopwords)
seq = 0
'''
Without Roughsets
All: 65%
1st 100: 85%
2nd 100: 70%
3rd 100: 55%
4th 100: 60%
5th 100: 70%
6th 100: 65%
7th 100: 50%

All without Normalization: 61%
'''
for i in range(0, 600):
    complaint = df.loc[i][0]
    complaint = complaint.translate(str.maketrans('', '', string.punctuation))
    tokens = scnlp.word_tokenize(complaint)
    newTokens = []

    tokens = ' '.join([normalize(str(f[0])) for f in scnlp.pos_tag(complaint) if f[1] != 'NNP' and f[1] != 'DTNNP' and str(f[0]).isalpha() == True and f[0] not in stopwords  ])
    preprocessed.loc[seq] = [tokens, df.loc[i][1]]
    seq += 1
    print(tokens)

cv = TfidfVectorizer()
X = cv.fit_transform(preprocessed['Title'])
y = preprocessed['Category']
y = encoder.fit_transform(y).tolist()
y = np.array(y)
print(X.shape)
X = X.toarray().tolist()
X = np.array(X)
#y = np.resize(y,(50, 363))


print(type(X))
print(type(y))

print(y)
#for xx in X:
#    print(xx)
selector = RoughSetsSelector()
#X_selected = selector.fit(X,y)
#X_selected = X_selected.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(20,20,20))
mlp.fit(x_train,y_train)
print('Prediction')
score = mlp.score(x_test, y_test)
print(score)