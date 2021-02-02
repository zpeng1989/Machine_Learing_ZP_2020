import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(0)

number_of_features = 1000

(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words = number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)
feature_train = tokenizer.sequences_to_matrix(data_train, mode = 'binary')
feature_test = tokenizer.sequences_to_matrix(data_test, mode = 'binary')

network = models.Sequential()

network.add(layers.Dense(units = 16, activation='relu',input_shape=(number_of_features,)))
network.add(layers.Dense(units = 16, activation='relu'))
network.add(layers.Dense(units = 1, activation='sigmoid'))

network.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath='model_best.h5', monitor='val_loss', save_best_only=True)]
history = network.fit(feature_train, target_train, epochs = 10, callbacks= callbacks,verbose=1, batch_size=100, validation_data=(feature_test, target_test))

