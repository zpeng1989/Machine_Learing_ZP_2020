import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow.keras import models

np.random.seed(0)

number_of_features = 5000

data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data
tokenizer = Tokenizer(num_words = number_of_features)
feature_train = tokenizer.sequences_to_matrix(data_train, mode = 'binary')
feature_test = tokenizer.sequences_to_matrix(data_test, mode = 'binary')

print(feature_train.shape)
print(target_vector_train[1])

target_train = to_categorical(target_vector_train)
target_test = to_categorical(target_vector_test)

network = models.Sequential()

network.add(layers.Dense(units=100, activation='relu', input_shape=(number_of_features,)))
network.add(layers.Dense(units=100, activation='relu'))
network.add(layers.Dense(units=46, activation='softmax'))
network.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
history = network.fit(feature_train, target_train, epochs = 10, verbose = 1, batch_size = 100, validation_data = (feature_test, target_test))









