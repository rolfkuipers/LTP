from __future__ import print_function
import numpy as np
import pickle as pk
import time
from random import shuffle
from data_class import User_Information
import sys
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


def mentions(tweets):
	counts = 0
	for tweet in tweets:
		if 'AT_USER' in tweet.split():
			counts += 1

	return counts/len(tweets)

def links(tweets):
	counts = 0
	for tweet in tweets:
		if 'URL' in tweet.split():
			counts += 1

	return counts/len(tweets)

def specials(tweets):
	total_count = 0
	specials = ['!','.',',',':','?',';','-','+','@',':)',':-)',':(',':-(','=']
	for tweet in tweets:
		count = 0
		for ch in specials:
			count += tweet.count(ch)
		total_count += count

	return total_count/len(tweets)

def get_data(user_list):
	total = []
	#classes = []
	for user in user_list:
		vector = []
		tweets = user.get_processed_tweets()
		vector.append(round(user.get_Average_length(),2))
		vector.append(round(mentions(tweets),2))
		vector.append(round(links(tweets),2))
		vector.append(round(specials(tweets),2))
		total.append((vector,user.get_Mbti()[0]))

	return total

start_time = time.time()

max_words = 1000
batch_size = 10
nb_epoch = 5

print('Loading dataset with pickle...')
user_list = pk.load(open("alluserlist.pickle", 'rb'))
print('Loading the data took: ', time.time() - start_time, 'seconds')

#total = get_data(user_list)


num_tweets = 100
num_users = 200

print('Loading tweets...')
all_tweets = []
combi = []
for user in user_list[:num_users]:
	joined_tweets = " ".join(user.get_processed_tweets()[:num_tweets])
	all_tweets.append(joined_tweets)

	#combi.append((joined_tweets,user.get_Mbti()[0]))
	cl = user.get_Mbti()[1]
	if cl == 'N':
		combi.append((joined_tweets,1))
	else:
		combi.append((joined_tweets,2))

shuffle(combi)

print('Fitting text on tokenizer...')
tokenizer = Tokenizer(nb_words=max_words)
tokenizer.fit_on_texts(all_tweets)

#tokenizer = Tokenizer(nb_words=)

print('Split text into train and test...')
X = [tweets for tweets,_ in combi]
y = []
for _, cl in combi:
	if cl == 'E':
		y.append(1)
	else:
		y.append(2)
#y = [cl for _,cl in combi]

#nb_classes = np.max(y_train)+1
#X = tokenizer.texts_to_sequences(X)
#X = tokenizer.sequences_to_matrix(X)
#y = np.utils.to_categorical(y,nb_classes)
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

"""
print('Building model... 1')
model = Sequential()

# DENSE
model.add(Dense(512, input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(max_words,)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=10, batch_size=5,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print()
print()

print('Building model... 2')
model = Sequential()

# DENSE
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print()
print()

print('Building model... 3')
model = Sequential()

# DENSE
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(max_words,)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print()
print()

print('Building model... 4')
model = Sequential()

# DENSE
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(max_words,)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print()
print()
print('Building model... 5')

model = Sequential()
# DENSE
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=10, batch_size=100	,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


"""
print('Building model... 6')
model = Sequential()

# DENSE
input_size = len(max(X_train, key=len))

X_train = sequence.pad_sequences(X_train, maxlen=input_size)
X_test = sequence.pad_sequences(X_test, maxlen=input_size)

model.add(Embedding(input_dim=max_words, output_dim=128, input_length=input_size, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

#model.add(Embedding(output_dim=128, input_dim=max_words, input_length=input_size))
#model.add(LSTM(output_dim=128, activation='softmax', inner_activation='tanh'))
#model.add(Dropout(0.2))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    nb_epoch=10, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

sys.exit()