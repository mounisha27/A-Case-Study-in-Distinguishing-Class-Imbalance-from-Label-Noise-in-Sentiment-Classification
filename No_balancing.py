import pandas as pd
import numpy as np

#read csv file
amazon_reviews_read = pd.read_csv('data.csv')

#change column name
amazon_reviews = amazon_reviews_read.rename(columns = {"reviews.text":"reviews",
"reviews.username":"username"})

#drop unwanted columns
to_drop = ['dateAdded','dateUpdated','name','asins','brand','categories','primaryCategories',
'imageURLs','keys','manufacturer','manufacturerNumber','reviews.date','reviews.dateAdded',
'reviews.dateSeen','reviews.doRecommend','reviews.id','reviews.numHelpful','reviews.rating',
'reviews.sourceURLs','reviews.title','sourceURLs' ]
amazon_reviews.drop(to_drop, inplace=True, axis=1)

#check changes
print(amazon_reviews.head(5))

import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

#remove punctuation
def remove_punctuations(reviews):
    for punctuation in string.punctuation:
        reviews = reviews.replace(punctuation, '')
    return reviews
amazon_reviews['reviews'] = amazon_reviews['reviews'].apply(remove_punctuations)


#tokenize
from nltk.tokenize import word_tokenize
amazon_reviews['reviews'] = amazon_reviews['reviews'].astype(str)
amazon_reviews['reviews'] = amazon_reviews['reviews'].apply(word_tokenize)

#stemming
from nltk.stem import LancasterStemmer
my_stemmer = LancasterStemmer()
amazon_reviews['reviews'] = [[my_stemmer.stem(word) for word in sentence] for sentence in amazon_reviews.reviews]

#remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
amazon_reviews['reviews'] = amazon_reviews['reviews'].apply(lambda x: [item for item in x if item not in stop]) 

#calculate sentiment_score
from textblob import TextBlob
def senti_pol(tokens):
    """Fixed version — scores the whole review."""
    full_review = " ".join(tokens)
    return TextBlob(full_review).sentiment.polarity

amazon_reviews['senti_polarity'] = amazon_reviews['reviews'].apply(senti_pol)
 
#dividing sentiments from sentiment polarity score
condition = [
        (amazon_reviews['senti_polarity'] > 0.05),
        (amazon_reviews['senti_polarity'] <= 0.05) & (amazon_reviews['senti_polarity'] > -0.05),
        (amazon_reviews['senti_polarity'] <= -0.05)
            ]
values = ['positive', 'neutral', 'negative']
amazon_reviews['sentiment'] = np.select(condition, values, default='Neutral')

print(amazon_reviews.head(5))

#check if classes are balanced
target_count = amazon_reviews.sentiment.value_counts()

# join tokens back into strings BEFORE splitting (needed for TfidfVectorizer)
amazon_reviews['reviews'] = [" ".join(review) for review in amazon_reviews['reviews'].values]

# define y before using it in the split
y = amazon_reviews['sentiment']

#vec
# 1. Split the RAW text and labels first — before any vectorizing or resampling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    amazon_reviews['reviews'], y, test_size=0.20, random_state=100, stratify=y
)

# 2. Fit the vectorizer ONLY on training text, then transform test text with it
#    (test text must never influence the vocabulary/IDF weights)
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()   # transform only, no fit

# 4. Train on the resampled training set, evaluate on the untouched real test set
from sklearn.svm import SVC
svclassifier = SVC()
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print(confusion_matrix(y_test, y_pred, labels=["negative", "positive", "neutral"]))
print("Accuracy for Support Vector Machine:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#NN model implementation
#NN model implementation
from sklearn.neural_network import MLPClassifier
clf_NN = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', solver='adam', random_state=1)
clf_NN.fit(X_train, y_train)   # <- was X_train, y_train
y_pred = clf_NN.predict(X_test)

#Chech NN model performance
from sklearn import metrics
print("Accuracy for Neural Network:", metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred, labels=["negative", "positive", "neutral"])
print(cm)
#svm model performance
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
print(classification_report(y_test,y_pred))