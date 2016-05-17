import pickle
import time
import re
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#from nltk.corpus import stopwords
from nltk import FreqDist

start_time = time.time()
print('loading pickle data....')
userlist = pickle.load(open("alluserlist.pickle", 'rb'),encoding='utf8')
#word_pos_pol = pickle.load(open('word_pos_pol.pickle', 'rb'))

#print('loading tweets and mbti and shuffle....')
print('Calculating most common words of total data set')
#stopwords = stopwords.words('dutch')
# First append everything as tupples to one list so the data can be shuffled.
totallist = []
for user in userlist:
	#m = user.get_Mbti()
	tweets = user.get_processed_tweets()
	#freq = FreqDist(" ".join(tweets))
	for tweet in tweets:
		totallist += tweet.split()
	#tweets = " ".join([word for word in tweets.split() if word not in freq.keys()])
	#total.append((" ".join(user.get_processed_tweets()),m))
	#total.append((" ".join(tweets),m))
	#for tweet in user.get_processed_tweets():
		#total.append((tweet,m))
#shuffle(total)
freq = FreqDist(totallist).most_common(50)
common_words = [word[0] for word in freq]
common_words.remove('AT_USER')
common_words.remove('URL')
#print(common_words)
#print(string)

print('Create total list of tweets and classes without common words')
total = []
for user in userlist:
	m = user.get_Mbti()[3]
	tweets = " ".join(user.get_processed_tweets()).split()
	#for tweet in tweets:
	tweets = [word for word in tweets if word not in common_words]
	total.append((" ".join(tweets),m))

shuffle(total)



print('create tweet and class lists of shuffled data....')
# Create total tweet and classes list respectively from the total tweet list
totaltweets = []
totalclasses = []
for tweets, m in total:
	totaltweets.append(tweets)
	totalclasses.append(m)

print('Split the data into train - dev and test sets....')
# Split the list in training, developer and test set 
split_point_1 = int(0.80*len(total))
split_point_2 = int(0.90*len(total))
X_train, X_dev, X_test = totaltweets[:split_point_1], totaltweets[split_point_1:split_point_2], totaltweets[split_point_2:]
Y_train, Y_dev, Y_test = totalclasses[:split_point_1], totalclasses[split_point_1:split_point_2], totalclasses[split_point_2:]

print('combining train and dev for using all data.... + test on Introvert-Extravert')
# Skip the developerset for a bit and set the class to only 1 binary class (for example Introvert-Extravert) instead of the combination
combiX_train = X_train + X_dev
combiY_train = Y_train + Y_dev
for i in range(len(combiY_train)):
	combiY_train[i] = combiY_train[i][0]
for i in range(len(Y_test)):
	Y_test[i] = Y_test[i][0]


cf_time = time.time()

# Create a union of features
print('creating union of features .....')
unionOfFeatures = FeatureUnion([
								('normaltfidf', TfidfVectorizer()),
								('bigrams', TfidfVectorizer(ngram_range = (2,2), analyzer = 'char')),
								('counts', CountVectorizer())
								])

#unionOfFeatures = FeatureUnion([
#								('normaltfidf', TfidfVectorizer()),
#								('counts', CountVectorizer())
#								])

#classifier = Pipeline([('vec', TfidfVectorizer()), ('cls', svm.SVC(kernel='rbf', C=1.5))])
print('fitting unions .....')
featureFit = unionOfFeatures.fit(combiX_train, combiY_train).transform(combiX_train)
print('creating classifier ....')
#classifier = Pipeline([('featureunion', unionOfFeatures), ('cls', svm.SVC(kernel='linear', C=1.0))])
classifier = Pipeline([('featureunion', unionOfFeatures),('clf', BernoulliNB())])
classifier.fit(combiX_train, combiY_train)

yGuess = classifier.predict(X_test)
print(classification_report(Y_test, yGuess))
print(accuracy_score(Y_test, yGuess))
print("Classifier time: --- %s seconds ---" % (time.time() - cf_time))
print("Total time: --- %s seconds ---" % (time.time() - start_time))

