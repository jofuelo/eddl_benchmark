import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.datasets import cifar10
import sys
import pickle
import tensorflow as tf
from keras.utils import np_utils
from keras.optimizers import Adam
import time

gpu = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else True
if gpu:
	gpus = tf.config.experimental.list_physical_devices("GPU")
	tf.config.experimental.set_memory_growth(gpus[0], True)

sys_details = tf.sysconfig.get_build_info()
print("CUDA version:", sys_details["cuda_version"], "; CUDNN version:", sys_details["cudnn_version"])
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

bn = int(sys.argv[1]) == 1

def addbn(model, bn):
	if bn:
		model.add(BatchNormalization())

def block(nf, model, reps, bn, input_sh = None):
	for i in range(reps):
		if input_sh is None:
			model.add(Conv2D(filters=nf, kernel_size=(3,3), padding="same", kernel_initializer="glorot_uniform" if bn else "he_uniform"))
		else:
			model.add(Conv2D(input_shape=input_sh,filters=nf,kernel_size=(3,3),padding="same", kernel_initializer="glorot_uniform" if bn else "he_uniform"))
		addbn(model, bn)
		model.add(ReLU())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model = Sequential()

block(64, model, 2, bn, (32,32,3))
block(128, model, 2, bn)
block(256, model, 4, bn)
block(512, model, 4, bn)
block(512, model, 4, bn)

model.add(Flatten())

for i in range(2):
	model.add(Dense(units=4096, kernel_initializer="glorot_uniform" if bn else "he_uniform"))
	addbn(model, bn)
	model.add(ReLU())

model.add(Dense(units=10, activation="softmax", kernel_initializer="glorot_uniform" if bn else "he_uniform"))

opt = Adam(lr=0.00001, epsilon=1e-06)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

hist = model.fit(x_train, y_train,
	epochs=50 if gpu else 1,
	callbacks=[time_callback],
	batch_size=50,
	#validation_data=(x_test, y_test)
	)

print("Mean Time:", np.mean(time_callback.times))
if gpu and False:
	with open("results/keras/keras_vgg19_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
		pickle.dump(hist.history["accuracy"], f)
	with open("results/keras/keras_val_vgg19_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
		pickle.dump(hist.history["val_accuracy"], f)
	print(hist.history["accuracy"], hist.history["val_accuracy"])
