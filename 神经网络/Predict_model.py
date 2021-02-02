import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from tensorflow.keras import layers

np.random.seed(0)
number_of_features = 10000
(data_train, target_train),(data_test, target_test) = imdb.load_data(num_words=number_of_features)
tokenizer = Tokenizer(number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode = 'binary')
features_test = tokenizer.sequences_to_matrix(data_test, mode = 'binary')
network = models.Sequential()
network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1, activation='sigmoid'))

network.compile(loss = 'binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = network.fit(features_train, target_train, epochs = 10, verbose = 1, batch_size=100, validation_data=(features_test, target_test))

predicted_target = network.predict(features_test)
print(predicted_target[0])
print(predicted_target[1])


