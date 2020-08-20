import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer



from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.models import model_from_json


from keras.layers import Embedding
from keras.layers import GlobalMaxPool1D
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Bidirectional


from keras.utils import to_categorical
from keras.utils import plot_model

import sys

########################################################

# 	PARAMS
# Corpus to classify.
corpus = str(sys.argv[1])

# Embeddings to use
embeddings = str(sys.argv[2])

# Neural network
nn = str(sys.argv[3])

########################################################


# VARS
batch_size = 32
epochs = 10
maxlen = 80
embedding_dim = 300


if (embeddings == 'fasttext'):
    # @var embedding_dim int The size of the word embeddings
    embedding_dim = 300
    # @var pretrained_embeddings String
    pretrained_embeddings_filename = './../pretrained_models/cc.es.300.vec'
    # @var pretrained_embeddings_encoding String
    pretrained_embeddings_encoding = "utf-8"

if (embeddings == 'glove'):
    # @var embedding_dim int The size of the word embeddings
    embedding_dim = 100
    # @var pretrained_embeddings String
    pretrained_embeddings_filename = './../../glove/model.txt'  
    # @var pretrained_embeddings_encoding String
    pretrained_embeddings_encoding = "utf-8"

if (embeddings == 'wiki'):
    # @var embedding_dim int The size of the word embeddings
    embedding_dim = 100    
    # @var pretrained_embeddings String
    pretrained_embeddings_filename = './../pretrained_models/wiki.es.vec'
    # @var pretrained_embeddings_encoding String
    pretrained_embeddings_encoding = "utf-8"


if (embeddings == 'embeddings-l-model'):
    # @var embedding_dim int The size of the word embeddings
    embedding_dim = 300
    # @var pretrained_embeddings String
    pretrained_embeddings_filename = './../pretrained_models/embeddings-l-model.vec'
    # @var pretrained_embeddings_encoding String
    pretrained_embeddings_encoding = "utf-8"



########################################################


def create_embedding_matrix (filepath, word_index, embedding_dim, enconding):
    
    # Adding again 1 because of reserved 0 index
    vocab_size = len (word_index) + 1  
    embedding_matrix = np.zeros ((vocab_size, embedding_dim))
  
    with open (filepath,'r', errors='ignore', encoding=enconding) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array (vector, dtype=np.float32)[:embedding_dim]


    return embedding_matrix


########################################################
# Read text
print ("Reading tweets from corpus "+corpus+".csv...")
df_texts_training = pd.read_csv ("../tweets/"+corpus+".csv", names=['sentence', 'label'], sep='\t')


print(df_texts_training)

# Reshape and normalize training data
print ("Get sentence features...")
sentences_features = df_texts_training['sentence'];

print(sentences_features)


sentences_labels = df_texts_training['label'];

print(sentences_labels)
##################Aqui empieza kfold

skf = StratifiedKFold (n_splits = 10)
total_actual = []
total_predicted = []
total_val_accuracy = []
total_val_loss = []
total_test_accuracy = []




file = open("../output/results-"+corpus+"-"+embeddings+"-"+nn+".txt","w")


for i, (train_index, test_index) in enumerate (skf.split (sentences_features,sentences_labels)):



	# Divide the sets
	print ("Divide text split")
	sentences_features_train, sentences_features_test, sentences_labels_train, sentences_labels_test = train_test_split (
		sentences_features,
		sentences_labels,
		test_size=0.2,
		random_state=60
	)

	sentences_features_train, sentences_features_test =		sentences_features.iloc[train_index], sentences_features.iloc[test_index]
	sentences_labels_train, sentences_labels_test = sentences_labels.iloc[train_index],	 sentences_labels.iloc[test_index]
		

	# Transform to binary
	print ("Binarize labels data...")
	lb = LabelBinarizer ()
	sentences_labels_train = lb.fit_transform (sentences_labels_train)
	sentences_labels_test = lb.fit_transform (sentences_labels_test)
	print(sentences_labels_test)

	# Create the tokenizer
	tokenizer = Tokenizer (num_words=5000)


	# Get tokens from training
	print ("Tokenize tweets...")
	tokenizer.fit_on_texts (df_texts_training['sentence'])


	# Get features for training and test
	sentences_features_train = tokenizer.texts_to_sequences (sentences_features_train)
	sentences_features_test = tokenizer.texts_to_sequences (sentences_features_test)



	# Get the vocab size
	vocab_size = len (tokenizer.word_index) + 1


	# Padding
	print ("Generate padding...")
	sentences_features_train = pad_sequences (sentences_features_train, maxlen=maxlen)
	sentences_features_test = pad_sequences (sentences_features_test, maxlen=maxlen)
	print (' --> text_features shape:', sentences_features_train.shape)
	print (' --> test_features_shape:', sentences_features_test.shape)



	print ("Generate embedding matrix with "+embeddings+"...")
	embedding_matrix = create_embedding_matrix (pretrained_embeddings_filename, tokenizer.word_index, embedding_dim, pretrained_embeddings_encoding)
	print (embedding_matrix)
	np.savetxt('./../matrix/'+corpus+"-"+embeddings+"-"+nn+'-matrix.txt',embedding_matrix)

	print ("Get how well is represented...")
	nonzero_elements = np.count_nonzero (np.count_nonzero (embedding_matrix,
		axis=1))
	print (nonzero_elements / vocab_size)
				
	main_embeddings_input = Input (shape=(None,))
	main_embeddings_1 = Embedding (vocab_size, embedding_dim, 
		weights=[embedding_matrix], trainable=True)(main_embeddings_input)
				
	print (main_embeddings_1)

	#OPCION1 = CONV1, LSTM, BI_LSTM
	# Bidirectional
		
	if (nn == 'bi'): main_embeddings_2 = Bidirectional(LSTM (embedding_dim, dropout=0.2, recurrent_dropout=0.2)
		)(main_embeddings_1)
	
	elif (nn == 'lstm'): main_embeddings_2 = LSTM (embedding_dim, dropout=0.2, recurrent_dropout=0.2)(main_embeddings_1)

	else : 
		output_1 = Conv1D (128, 5, activation='relu')(main_embeddings_1)
		output_2 = GlobalMaxPool1D ()(output_1)
		main_embeddings_2 = Dense (10, activation='relu')(output_2)
			
		
	#OPCION2 = SIGMOID, SOFTMAX
	main_embeddings_3 = Dense (10, activation='softmax')(main_embeddings_2)
			
	merged = main_embeddings_3


	# Final predictions
	hidden1 = Dense(10, activation='relu')(merged)
	hidden2 = Dense(10, activation='relu')(hidden1)

	predictions = Dense (1, activation='sigmoid')(hidden2)



	#Create model
	model = Model (inputs=main_embeddings_input, outputs=predictions)

	# Compile model
	print ("Compile model...")
	model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	
	print (model.summary ())	
	
	hist = model.fit (
			sentences_features_train,
			sentences_labels_train,
			batch_size=batch_size,
			epochs=15,
			validation_data=(
			sentences_features_test,
			sentences_labels_test)
		)
	score, acc = model.evaluate (
		sentences_features_test,
		sentences_labels_test, 
		batch_size=batch_size)


	file.write('\nTest score: '+ str(score))
	file.write('\nTest accuracy: '+ str(acc))

	total_val_accuracy.append (acc)
	
print (total_val_accuracy)
print ("Mean validation accuracy: {}%".format (np.mean (total_val_accuracy) * 100))

print (total_test_accuracy)
print ("Mean validation accuracy: {}%".format (np.mean (total_test_accuracy) * 100))


file.write('\n'+str(total_val_accuracy))
file.write("Mean validation accuracy: {}%".format (np.mean (total_val_accuracy) * 100))

file.write('\n'+str(total_test_accuracy))
file.write("Mean validation accuracy: {}%".format (np.mean (total_test_accuracy) * 100))

file.close()

	
