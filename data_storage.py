from data_class import User_Information
import json
import os
import pickle
import re

def data_storage(filename):
	data = json.load(open(filename),encoding='utf8')
	user_list = []
	for key, values in data.items():
		user_list.append(User_Information(key, values))

	return user_list

def main():
	filename = "data/TwiSty-NL.json"
	user_list = data_storage(filename)

	count = 0
	for root, _, files in os.walk('data/users_id/'):
		for f in files:
			data = json.load(open(os.path.join(root, f)),encoding='utf8')
			for user in user_list:
				if user.get_ID() == f.replace('.json', ""):
					user.set_Tweets(data)
					count += 1
			if count % 50:
				print(count)

	pickle.dump(user_list, open('user_data.pickle', 'wb'))

main()	
