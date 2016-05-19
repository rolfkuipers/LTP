from data_class import User_Information
import pickle

def process_data(user_data, argument):
	# Load the 6000 most informative words for the argument data
	with open('informative_words/'+ argument + '-information.txt','r') as f:
		information = f.readlines()
		information = [x.strip('\n') for x in information]

	# Process the tweets regarding the most informative words. Only keep words if they
	# are part of the 600. The [:500] index means only to use 500 tweets, same as
	# used in rainbow to get the 6000 most informative words. 
	count = 0
	informative_data = []
	for user in user_data:
		tweets = " ".join(user.get_processed_tweets()[:500])
		tweets = " ".join([word for word in tweets.split() if word in information])
		informative_data.append((tweets,user.get_Mbti()[int(argument)]))
		print(count)
		count += 1


	return informative_data

def main():
	print("Loading pickled user data ......")
	user_data = pickle.load(open('user_data.pickle','rb'))

	print("Process user data ...")
	# The second parameter 0 means which class to use. 0 is Introvert/Extravert, corresponding to the
	# first position in the personality abbrevation
	processed_user_data = process_data(user_data,'3')

	# Change name to the correct pickle file name
	pickle.dump(processed_user_data,open('processed_judging_perceiving.pickle', 'wb'))

main()