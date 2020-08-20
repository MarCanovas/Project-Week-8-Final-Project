
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


full = pd.read_csv ("../tweets/full_corpus.csv", names=['sentence', 'label'], sep='\t')

print(full)

news = pd.read_csv ("../tweets/news.csv", names=['sentence', 'label'], sep='\t')

news.label = 'non-misogyny'

print(news)

data = pd.concat([full,news])

print(data)
data.to_csv('../tweets/balanced-corpus.csv', index=False, header=False, sep='\t')


