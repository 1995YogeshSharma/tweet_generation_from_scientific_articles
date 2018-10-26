from collections import defaultdict

import constants
import process_dataset
import generate_models
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

def get_term_id_dict(f):
	term_id_dict = defaultdict(int)
	term_id_rdict = defaultdict(str)
	
	term_id = 1
	count = 0
	for line in f.readlines():
		print line.strip().split()
		for word in line.strip().split():
			if word.strip() not in term_id_dict.keys():
				#print 'creation '
				#print word, term_id
				term_id_dict[word.strip()] = term_id
				term_id_rdict[term_id] = word.strip()
				term_id += 1
			if count > 1000:
				break
			count += 1
	return (term_id_dict, term_id_rdict)

if __name__ == '__main__':

	article_term_id_dict, article_term_id_rdict = get_term_id_dict(open(constants.article_file, 'r'))
	tweet_term_id_dict, tweet_term_id_rdict = get_term_id_dict(open(constants.tweet_file, 'r'))

	X1, X2, y = process_dataset.get_dataset(article_term_id_dict, tweet_term_id_dict)

	print 'printing x1'
	print X1[0:5]
	print 'printing x2'
	print X2[0:5]
	print 'printing y'
	print y[0:5]

	train_model, inf_enc_model, inf_dec_model = generate_models.get_models(
		500, 100, 128)
	print 'got all the models...'
	train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	print 'model compiled...'
	train_model.fit([X1, X2], y, epochs=5, verbose=1)

	# evaluating LSTM 
	

