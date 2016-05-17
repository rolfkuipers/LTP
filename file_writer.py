#import time
from nltk.corpus import stopwords
import pickle
from random import shuffle

#start_time = time.time()
print('loading pickle data....')
userlist = pickle.load(open("alluserlist.pickle", 'rb'),encoding='utf8')

print('loading tweets and mbti and shuffle....')
stopwords = stopwords.words('dutch')
# First append everything as tupples to one list so the data can be shuffled.
total = []
for user in userlist[:50]:
	m = user.get_Mbti()
	tweets = user.get_processed_tweets()[:100]
	for tweet in tweets:
		tweet = " ".join([word for word in tweet.split() if word not in stopwords])
	#total.append((" ".join(user.get_processed_tweets()),m))
	total.append((" ".join(tweets),m[3]))
	#for tweet in user.get_processed_tweets():
		#total.append((tweet,m))
shuffle(total)

#print('create tweet and class lists of shuffled data....')
# Create total tweet and classes list respectively from the total tweet list
totaltweets = []
totalclasses = []
for tweets, m in total:
	totaltweets.append(tweets)
	totalclasses.append(m)

#print('Split the data into train - dev and test sets....')
# Split the list in training, developer and test set 
#split_point_1 = int(0.80*len(total))
#split_point_2 = int(0.90*len(total))
#X_train, X_dev, X_test = totaltweets[:split_point_1], totaltweets[split_point_1:split_point_2], totaltweets[split_point_2:]
#Y_train, Y_dev, Y_test = totalclasses[:split_point_1], totalclasses[split_point_1:split_point_2], totalclasses[split_point_2:]

#split_point_1 = int(0.80*len(total))
split_point_2 = int(0.90*len(total))
X_train, X_test = totaltweets[:split_point_2], totaltweets[split_point_2:]
Y_train, Y_test = totalclasses[:split_point_2], totalclasses[split_point_2:]

extra_train = open('extra_train','w',encoding='utf8')
intro_train = open('intro_train','w',encoding='utf8')
extra_test = open('extra_test','w',encoding='utf8')
intro_test = open('intro_test','w',encoding='utf8')
#dev = open('judper_dev.tsv','w',encoding='utf8')
#test = open('judper_test.tsv','w',encoding='utf8')

for i in range(len(X_train)):
	if Y_train[i] == 'E':
		extra_train.write(X_train[i] + '\n')
	else:
		intro_train.write(X_train[i] + '\n')
for i in range(len(X_test)):
	if Y_test[i] == 'E':
		extra_test.write(X_test[i] + '\n')
	else:
		intro_test.write(X_test[i] + '\n')

#for i in range(len(X_dev)):
#	dev.write(X_dev[i] + '\t' + Y_dev[i]+ '\n')
#for i in range(len(X_test)):
#	test.write(X_test[i] + '\t' + Y_test[i]+ '\n')

#train.close()
#dev.close()
#test.close()

extra_train.close()
intro_train.close()
extra_test.close()
intro_test.close()