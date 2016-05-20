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
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

start_time = time.time()

# Setting sparameters
max_words = 6000

parser = argparse.ArgumentParser()
parser.add_argument('--type',help="which classification problem to use", choices=("ie","si","ft", "jp"), default="ie")
parser.add_argument('--NN', help="type of NN", choices=("lstm","simplernn",'dense'), default="simplernn")
args = parser.parse_args()

pickles = {'ie':'introvert_extravert','si':'sensing_intuition','ft':'feeling_thinking','jp':'judging_perceiving'}
print('loading pickle data....')
user_list = pickle.load(open('processed_'+ pickles[args.type] + '.pickle', 'rb'))
shuffle(user_list)

X, y   = [], []
for tweets, m in user_list:
	if len(tweets) < 1:	pass
	else: X.append(tweets)
		if args.type[0].upper() == m: y.append(1)
		else: y.append(2)

# Create a tokenizer
print('Fitting text on tokenizer...')
tokenizer = Tokenizer(nb_words=max_words)
tokenizer.fit_on_texts(X)

# Split the data
print('Split text into train and test...')

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


# Pad input sequences
input_size = len(max(X_train, key=len))

X_train = sequence.pad_sequences(X_train, maxlen=input_size)
X_test = sequence.pad_sequences(X_test, maxlen=input_size)

# Setting some parameters
batch_size = 20
nb_epoch = 5

print('Building model... 6')
model = Sequential()

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
