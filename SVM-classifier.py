""" Rolf Kuipers	s2214415	Rijksuniversiteit Groningen
	SVM Classifier for Language Technology Project

	USAGE VIA COMMAND LINE:
	python svm-classifier.py --type sn
	## type can be one of ie (introvert-extravert), sn (sensing-intuition), 
	## ft (feeling-thinking), jp (judging-perceiving).
"""

import pickle
import time
import re
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
import argparse

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--type',help="which classification problem to use", choices=("ie","sn","ft", "jp"), default="ie")
args = parser.parse_args()

pickles = {'ie':'introvert_extravert','sn':'sensing_intuition','ft':'feeling_thinking','jp':'judging_perceiving'}
print('loading pickle data....')
user_list = pickle.load(open('processed_'+ pickles[args.type] + '.pickle', 'rb'))

# Shuffle list (tweet, class) tuples 
shuffle(user_list)

print('create tweet and class lists of shuffled data....')
totaltweets = []
totalclasses = []
for tweets, m in user_list:
	totaltweets.append(tweets)
	totalclasses.append(m)

print('Split the data into train and test set....')
split_point_1 = int(0.90*len(user_list))
X_train, X_test = totaltweets[:split_point_1], totaltweets[split_point_1:]
Y_train, Y_test = totalclasses[:split_point_1], totalclasses[split_point_1:]

# Create a union of features
print('creating union of features .....')
unionOfFeatures = FeatureUnion([
								('normaltfidf', TfidfVectorizer()),
								('bigrams', TfidfVectorizer(ngram_range = (2,2), analyzer = 'char')),
								('counts', CountVectorizer())
								])

print('fitting unions .....')
featureFit = unionOfFeatures.fit(X_train, Y_train).transform(X_train)
print('creating classifier ....')
classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.0))])
classifier.fit(X_train, Y_train)

yGuess = classifier.predict(X_test)
print(classification_report(Y_test, yGuess))
print(accuracy_score(Y_test, yGuess))
print("Total time: --- %s seconds ---" % round((time.time() - start_time),2))

