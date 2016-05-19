from __future__ import print_function
import numpy as np
import pickle 
import time
from random import shuffle
from data_class import User_Information
import sys
import argparse
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

start_time = time.time()

# Setting some parameters
max_words = 6000


num_tweets = 500	
num_users = 1000	# Is all users

# Loading the data from pickle (over 1GB, takes some time)

parser = argparse.ArgumentParser()
parser.add_argument('--type',help="which classification problem to use", choices=("ie","si","ft", "jp"), default="ie")
parser.add_argument('--NN', help="type of NN", choices=("lstm","simplernn",'dense'), default="simplernn")
args = parser.parse_args()

pickles = {'ie':'introvert_extravert','si':'sensing_intuition','ft':'feeling_thinking','jp':'judging_perceiving'}
print('loading pickle data....')
user_list = pickle.load(open('processed_'+ pickles[args.type] + '.pickle', 'rb'))
shuffle(user_list)
#print('Loading dataset with pickle...')
#user_list = pk.load(open("alluserlist.pickle", 'rb'))
#print('Loading the data took: ', time.time() - start_time, 'seconds')
"""
# Load all the twitter data
print('Loading tweets...')
all_tweets = []
combi = []
for user in user_list[:num_users]:
	joined_tweets = " ".join(user.get_processed_tweets()[:num_tweets])
	all_tweets.append(joined_tweets)

	# The index [0] and cl == 'E' have to be changed to test another class (e.g. Judging - Percieving).
	# Then the index would be [3] and cl == 'J' 
	cl = user.get_Mbti()[0]
	if cl == 'E':
		combi.append((joined_tweets,1))
	else:
		combi.append((joined_tweets,2))

# Shuffle the data
shuffle(combi)
"""
X = []
y = []
for tweets, m in user_list:
	if len(tweets) < 1:
		pass
	else:
		X.append(tweets)
		if args.type[0].upper() == m:
			y.append(1)
		else:
			y.append(2)

# Create a tokenizer
print('Fitting text on tokenizer...')
tokenizer = Tokenizer(nb_words=max_words)
tokenizer.fit_on_texts(X)

# Split the data
print('Split text into train and test...')
#X = [tweets for tweets,_ in user_list]
#y = [cl for _, cl in user_list]

split_point = int(len(X) * 0.90)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print('Text to sequence - sequence to matrix for data ...')
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = tokenizer.sequences_to_matrix(X_train)
X_test = tokenizer.sequences_to_matrix(X_test)

nb_classes = np.max(y_train)+1
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)



# LSTM model
input_size = len(max(X_train, key=len))

X_train = sequence.pad_sequences(X_train, maxlen=input_size)
X_test = sequence.pad_sequences(X_test, maxlen=input_size)
print('Building model... 6')
model = Sequential()
if args.NN == 'lstm':
	batch_size = 20
	nb_epoch = 5
	model.add(Embedding(input_dim=max_words, output_dim=128, input_length=input_size, dropout=0.2))
	model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid'))

if args.NN == 'simplernn':
	batch_size = 20
	nb_epoch = 5
	model.add(Embedding(input_dim=max_words, output_dim=128, input_length=input_size, dropout=0.2))
	model.add(SimpleRNN(128, dropout_W=0.2, dropout_U=0.2))  
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid'))

if args.NN == 'dense':
	batch_size = 20
	nb_epoch = 5
	model.add(Dense(512, input_shape=(max_words,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

sys.exit()
