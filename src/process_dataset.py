import pickle
import numpy as np
from keras.utils import to_categorical

import constants

article_file = constants.article_file
tweet_file = constants.tweet_file

def get_dataset(article_term_id_dict, tweet_term_id_dict):
	f_article = open(article_file, 'r')
	f_tweet = open(tweet_file, 'r')
	
	# The data in both input files is tokenized
	d_article = f_article.read().strip().split('\n')
	d_article = [row.split() for row in d_article]
	# converting all terms to term ids
	# print 'length of article file ', len(d_article)
	# NOTE: replace 1000 by len(d_article)
	for i in range(1000):
		for j in range(len(d_article[i])):
			the_word = d_article[i][j]
			d_article[i][j] = article_term_id_dict[d_article[i][j].strip()]
			#if(d_article[i][j] == 0):
			#	print 'ye raha '
			#	print the_word

	# converting tweet term to term ids
	d_tweet = f_tweet.read().strip().split('\n')
	d_tweet = [row.split() for row in d_tweet]
	# print 'length of tweet file ', len(d_tweet)
	# NOTE: replace 1000 by len(d_tweet)
	for i in range(1000):
		for j in range(len(d_tweet[i])):
			d_tweet[i][j] = tweet_term_id_dict[d_tweet[i][j]]

	# Now each row in data lists correspond to article/tweet

	# initializing the data vectors
	X1, X2, y = list(), list(), list()
	
	size_check = None
	for i in range(1000):
		article = d_article[i][:] + [0]*(500 - len(d_article[i][:]))#src
		tweet = d_tweet[i][:] + [0]*(100 - len(d_tweet[i][:]))#target
		tweet_in = [0] + tweet[:-1] #target_in
		article_encoded = to_categorical([article], num_classes=len(article_term_id_dict))
		if i < 10:
			print 'i '
			print len(article)
			print article_encoded.size
		if size_check is None:
			size_check = article_encoded.size
		if article_encoded.size != size_check:
			continue
			
		tweet_encoded = to_categorical([tweet], num_classes=len(tweet_term_id_dict))
		tweet_in_encoded = to_categorical([tweet_in], num_classes=len(tweet_term_id_dict))

		X1.append(article_encoded)
		X2.append(tweet_in_encoded)
		y.append(tweet_encoded)
		#print i
	#print X1[0:5]
	#print 'printing X2 '
	#print X2[0:5]
	#print 'printing y '
	#print y[0:5]
	#print 'size of x1 ', len(X1)
	#print X1[0]
	#print 'size of each vector in x1 ', X1[0].size
	return np.array(X1), np.array(X2), np.array(y)
	
