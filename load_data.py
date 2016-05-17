import pickle
from data_class import User_Information
#import requests
#from langdetect import detect
import time
#from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def main():
	start = time.time()
	print("loading user list...")
	user_list = pickle.load(open("alluserlist.pickle", 'rb'))
	print("writing files")
	stopword = stopwords.words('dutch')
	#tweets = ["de speakers moeten harder om de zwartelijst goed te horen.", "lijkt me heel anders radio maken dan je gewend bent"]
	for user in user_list:
		tweets = user.get_processed_tweets()
		for tweet in tweets:
			ind = tweets.index(tweet)
			tweets[ind] = " ".join([word for word in tweet.split() if word not in stopword])
		#if user.get_Mbti()[3] == 'J':
		if user.get_Gender() == 'M':
			f = open('male/' + user.get_ID(),'w',encoding='utf8')
			f.write(" ".join(tweets))
			f.close()
		else:
			f = open('female/' + user.get_ID(),'w',encoding='utf8')
			f.write(" ".join(tweets))
			f.close()

	#with open('allusers.json','w') as fp:
	
	#users = ujson.load(open('allusers.json','r'))
	#print(len(users))
	#print(user_list[0].get_processed_tweets())
	print('seconds: ', time.time() - start)
"""
	alltweets = []
	count = 0
	for user in user_list:
		for tweet in user.get_processed_tweets():
			alltweets.append(tweet)
		count += 1
		print(count)
	print(len(alltweets))

	#word_pos_pol = pickle.load(open('word_pos_pol.pickle', 'rb'))	

	#tknzr = TweetTokenizer()
"""
"""
	#tnt_pos_tagger = tnt.TnT()
	#tnt_pos_tagger.train(conll2002.tagged_sents('ned.train'))
	#posTagged = tnt_pos_tagger.tag(['ik', 'beb', 'een', 'lieve', 'jongen'])
	#print(posTagged)
	positive_NL = open('../positive_nl.txt', 'r').read().splitlines()
	negative_NL = open('../negative_nl.txt','r').read().splitlines()

	
	geen_niet_not = ['geen','niet','not']

	pos_tweets = 0
	neg_tweets = 0
	neutral_tweets = 0

	count = 0
	for user in user_list:
		for tweet in user.get_processed_tweets():
			try:
				neg_count = 0
				pos_count = 0
				tokens = tweet.split()
				if detect(tweet) == 'nl':
					for token in tokens:
						if token in positive_NL:
							# negatief voorafgaand aan positief (bijvoorbeeld 'niet mooi', is dus negatief)
							if tokens[tokens.index(token)-1] in geen_niet_not:
								neg_count += 1
							else:
								pos_count += 1
						if token in negative_NL:
							# negatief voorafgaand aan negatief (bijvoorbeeld 'niet slecht', is dus positief)
							if tokens[tokens.index(token)-1] in geen_niet_not:
								pos_count += 1
							if token in geen_niet_not and tokens[tokens.index(token)+1] not in positive_NL+negative_NL:
								continue
							else:
								neg_count += 1	

				if pos_count > neg_count:
					pos_tweets += 1
				if neg_count > pos_count:
					neg_tweets += 1
				if pos_count == neg_count:
					neutral_tweets += 1

				count += 1
				if count % 150 == 0:
					print('Processed... ' + str(count))
			except:
				continue


		print('Total tweets: ' + str(len(user.get_processed_tweets())))
		print('Positive tweets: ' + str(pos_tweets))
		print('Negative tweets: ' + str(neg_tweets))
		print('Neutral tweets: ' + str(neutral_tweets))
		print(pos_tweets + neg_tweets + neutral_tweets)
		pos_tweets, neg_tweets, neutral_tweets = 0,0,0



	
	url = 'http://text-processing.com/api/sentiment/'
	headers = {}
	for tweet in user_list[0].get_Tweet_texts():
		lang = detect(tweet)
		if lang == 'nl':
			print(tweet)
			print('Gegeven taal: Nederlands')
			payload = {'language': 'dutch', 'text': tweet}
			print(requests.post(url, data=payload, headers=headers).json()['probability'])
			print()
		if lang == 'en':
			print(tweet)
			print('Gegeven taal: Engels')
			payload = {'text': tweet}
			print(requests.post(url, data=payload, headers=headers).json()['probability'])
			print()
		else:
			print(tweet)
			print('Taal onbekend')
			payload = {'language': 'dutch', 'text': tweet}
			print(requests.post(url, data=payload, headers=headers).json()['probability'])
			print()
	
		


	

"""
main()