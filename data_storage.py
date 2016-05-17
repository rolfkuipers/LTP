from data_class import User_Information
import json
import os
import pickle
import xml.etree.ElementTree as ET
import re

def Data_Storage(filename):
	data = json.load(open(filename),encoding='utf8')
	user_list = []
	for key, values in data.items():
		user_list.append(User_Information(key, values))

	return user_list

def main():
	filename = "real_data/all_nl/TwiSty-NL.json"
	user_list = Data_Storage(filename)
	print(len(user_list))

	count = 0
	for root, _, files in os.walk('real_data/all_nl/nl/users_id/'):
		for f in files:
			data = json.load(open(os.path.join(root, f)),encoding='utf8')
			for user in user_list:
				if user.get_ID() == f.replace('.json', ""):
					user.set_Tweets(data)
					count += 1
			if count % 50:
				print(count)
	
	print(user_list[0].get_processed_tweets())

	#for user in user_list:
	#	print(user.get_Average_length())

	pickle.dump(user_list, open('alluserlist.pickle', 'wb'))
"""
	tree = ET.parse('../NL-sentiment-lexicon.xml')
	root = tree.getroot()
	sentence = "leuk is best wel gemeen"
	
	word_pos_pol = []

	for LexicalEntry in root.iter('LexicalEntry'):
		tup = []
		for Lemma in LexicalEntry.iter('Lemma'):
			tup.append(Lemma.attrib['writtenForm'])
		for Sense in LexicalEntry.iter('Sense'):
			for Sentiment in Sense.iter('Sentiment'):
				tup.append(Sentiment.attrib['polarity'])
		tup.append(LexicalEntry.attrib['partOfSpeech'])
		word_pos_pol.append(tup)

	pickle.dump(word_pos_pol, open('word_pos_pol.pickle','wb'))	
"""
main()	