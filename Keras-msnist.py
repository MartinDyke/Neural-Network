#import mnist dataset, as well as keras definitions to be used
from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

if __name__ == "__main__":
	#make a new sequential model
	network = models.Sequential()
	network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
	network.add(layers.Dense(10, activation='softmax'))
	network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	
	#Reshape training images from 600000,28,28 to 60000,28x28 and change data to be in the range [0,1]
	train_images = train_images.reshape((60000,28*28))
	train_images = train_images.astype('float32') / 255
	
	# Reshape test images from 600000,28,28 to 60000,28x28 and change data to be in the range [0,1]
	test_images = test_images.reshape(10000,28*28)
	test_images = test_images.astype('float32') / 255
	
	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)
	
	
	network.fit(train_images,train_labels,epochs=5, batch_size=128)