from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers

def vectorize_sequences(sequences, dim=10000):
	results = np.zeros((len(sequences), dim))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
		return results




if __name__ == "__main__":
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
	results = np.zeros((len(train_data), 10000))
	x_train = vectorize_sequences(train_data)
	x_test = vectorize_sequences(test_data)
	y_train = np.asarray(train_labels).astype('float32')
	y_test = np.asarray(test_labels).astype('float32')
	
	#Model to be sequential with 3 layers, inputs outputs to be [10000,16], [16,16], [16,1]
	#Activation of third layer to be sigmoidal in order to give a probability
	model = models.Sequential()
	model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	
	model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
	