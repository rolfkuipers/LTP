import re
import string
from nltk.tokenize import wordpunct_tokenize

def processTweet(tweet,printable):
	# Process the tweets
	
	# Convert to lower case
	tweet = tweet.lower()
	# Convert www.* or https?://* to URL
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	# Convert @username to AT_USER
	tweet = re.sub('@[^\s]+','AT_USER',tweet)
	# Remove additional white spaces
	tweet = re.sub('[\s]+', ' ', tweet)
	# Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	# trim
	tweet = tweet.strip('\'"')
	# tokenize on punctuation
	tweet = " ".join(wordpunct_tokenize(tweet))
	# remove personality abbreviations (e.g. ENFP etc.)
	traits = ['istj','isfj','infj','intj','istp','isfp','infp','intp','estp','esfp','enfp','entp','estj','esfj','enfj','entj']
	for word in tweet.split():
		if word in traits:
			tweet = " ".join([word for word in tweet.split() if word not in traits])

	# remove weird characters
	tweet = "".join(filter(lambda x: x in printable, tweet))

	return tweet

class User_Information:
	def __init__(self, ID, data):
		self.ID = ID
		self.data = data

		self.mbti = data['mbti']
		self.gender = data['gender']
		self.tweet_ids = data['other_tweet_ids']

	def get_ID(self):
		return self.ID

	def get_Mbti(self):
		return self.mbti

	def get_Gender(self):
		return self.gender

	def get_Tweet_ids(self):
		return self.tweet_ids

	def set_Tweets(self, tweet_data):
		self.all_tweet_info = tweet_data['tweets']
		all_tweets = []
		all_processed = []
		printable = set(string.printable)
		for key, value in self.all_tweet_info.items():
			all_tweets.append(value['text'])
			all_processed.append(processTweet(value['text'],printable))


		self.tweet_texts = all_tweets
		self.processed_tweets = all_processed

		lengths = [len(tweet) for tweet in self.tweet_texts]
		self.average_tweet_length = round(float(sum(lengths) / len(lengths)),2)

	def get_All_tweet_info(self):
		return self.all_tweet_info

	def get_Tweet_texts(self):
		return self.tweet_texts

	def get_processed_tweets(self):
		return self.processed_tweets

	def get_Average_length(self):
		return self.average_tweet_length




