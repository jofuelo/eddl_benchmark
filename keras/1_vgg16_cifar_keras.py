import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.datasets import cifar10
import sys
import pickle
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

bn = int(sys.argv[1]) == 1

def addbn(model, bn):
	if bn:
		model.add(BatchNormalization())

def block(nf, model, reps, bn, input_sh = None):
	for i in range(reps):
		if input_sh is None:
			model.add(Conv2D(filters=nf, kernel_size=(3,3), padding="same"))
		else:
			model.add(Conv2D(input_shape=input_sh,filters=nf,kernel_size=(3,3),padding="same"))
		addbn(model, bn)
		model.add(ReLU())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model = Sequential()

block(64, model, 2, bn, (32,32,3))
block(128, model, 2, bn)
block(256, model, 3, bn)
block(512, model, 3, bn)
block(512, model, 3, bn)

model.add(Flatten())

for i in range(2):
	model.add(Dense(units=4096))
	addbn(model, bn)
	model.add(ReLU())

model.add(Dense(units=10, activation="softmax"))



from keras.optimizers import Adam
opt = Adam(lr=0.0001, epsilon=1e-06)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

hist = model.fit(x_train, y_train,
	epochs=50,
	batch_size=100,
	validation_data=(x_test, y_test)
	)

with open("results/keras/keras_vgg16_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
	pickle.dump(hist.history["accuracy"], f)
with open("results/keras/keras_val_vgg16_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
	pickle.dump(hist.history["val_accuracy"], f)
print(hist.history["accuracy"], hist.history["val_accuracy"])