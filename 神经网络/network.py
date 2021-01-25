from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(units = 16, activation = 'relu', input_shape = (10,)))
network.add(layers.Dense(units = 16, activation = 'relu'))
network.add(layers.Dense(units = 1, activation = 'sigmoid'))

network.complie(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


print('sssss')
