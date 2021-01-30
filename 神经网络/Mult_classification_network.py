import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

np.random.seed(0)

number_of_features = 5000

data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data
tokenizer = Tokenizer(num_words = number_of_features)
feature_train = tokenizer.sequences_to_matrix(data_train, mode = 'binary')
feature_test = tokenizer.sequence_to_matrix(data_test, mode = 'binary')





