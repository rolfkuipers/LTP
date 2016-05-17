import random
from sklearn.metrics import classification_report
import pickle
import time
"""
class Perceptron:
	def __init__(self):
		self.self = self
		#self.n = n # length of array with weights
		#self.c = c # constant for error

	def feedforward(self, inputs, weights):
		sum = 0
		for i in range(len(inputs)):
			sum += inputs[i] * weights[i]
		self.sum = sum

	def sum(self):
		return self.sum

	def activate(self):
		if self.sum < 0:
			output = -1
		else:
			output = 1
		return output
"""

def mentions(tweets):
	counts = 0
	for tweet in tweets:
		for word in tweet.split():
			if word == 'AT_USER':
				counts += 1

	return round(counts/len(tweets),2)

def links(tweets):
	counts = 0
	for tweet in tweets:
		for word in tweet.split():
			if word == 'URL':
				counts += 1

	return round(counts/len(tweets),2)

def main():
	# input
	start_time = time.time()
	print('Loading user list...')
	user_list = pickle.load(open("alluserlist.pickle", 'rb'))
	print("Getting inputs...")
	inputs = []
	cf = []
	for user in user_list:
		vector = []
		tweets = user.get_processed_tweets()
		# append average tweet length
		vector.append(user.get_Average_length())
		# append average mentions
		vector.append(mentions(tweets))
		# append average links
		vector.append(links(tweets))
		# append polarity
		inputs.append(vector)
		cl = user.get_Mbti()[0]
		if cl == 'E':
			cf.append(-1)
		else:
			cf.append(1)

	#print(inputs)
	
	
	print("Creating classifier...")

	#inputs = [[1.3,2.4,64],[1.2,2.3,65],[2.3,2.9,100],[2.4,3.0,102],[1.6,2.2,74],[0.3,2.3,85],[3.3,2.3,120],[2.6,2.0,112]]
	#cf= [-1,-1,1,1,-1,-1,1,1]
	
	bias = 1
	for xy in inputs:
		xy.append(bias)

	weights = []
	for i in range(len(inputs)):
		weights.append([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])

	runs = 10
	for i1 in range(runs):
		result = []
		for i2 in range(len(inputs)):
			sums = 0
			for i3 in range(len(inputs[0])):
				sums += inputs[i2][i3] * weights[i2][i3]
			if sums < 0:
				guess = -1
			else:
				guess = 1
			if guess == cf[i2]:
				error = 0
			elif guess == 1 and cf[i2] == -1:
				error = -2
			else:
				error = + 2
			for x in range(len(weights[i2])):
				newWeight = weights[i2][x] + (error * inputs[i2][x] * 0.001)
				weights[i2][x] = newWeight
			result.append(guess)
		#print("True classification:", cf)
		#print("Guessed by classifier:", result, '\n')
		print(classification_report(cf,result))
		#print(weights[0])

	print("The program took: ", time.time() - start_time)

	#p = Perceptron()
	#p.feedforward(x, w)
	#print(p.sum())
	   

	#percep = Perceptron(3,0.001)
	#percep.weights()
	#percep.feedforward([4,12,6])
	#print(percep.activate())


   # verkrijg input output matrix
	#creeer random weigths matrix met zelfde lengte



main()


