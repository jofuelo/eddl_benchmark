import keras,os
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Add, Input, BatchNormalization, ReLU, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10
import sys
import pickle
import tensorflow as tf
from keras.optimizers import Adam
import time
from keras.utils import np_utils

gpu = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else True
if gpu:
	gpus = tf.config.experimental.list_physical_devices("GPU")
	tf.config.experimental.set_memory_growth(gpus[0], True)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

bn = int(sys.argv[1]) == 1

def batchn(bn, lc):
	if bn:
		return BatchNormalization()(lc)
	else:
		return lc


def resnet_block(l0, nf, bn, reps, downsample=True):
	for i in range(reps):
		l1 = Conv2D(filters=nf, kernel_size=(1,1), strides=((2,2) if (downsample and i==0) else (1,1)), padding="same", use_bias=False)(l0)
		l1 = batchn(bn, l1)
		l1 = ReLU()(l1)
		l1 = Conv2D(filters=nf, kernel_size=(3,3), padding="same", use_bias=False)(l1)
		l1 = batchn(bn, l1)
		l1 = ReLU()(l1)
		l1 = Conv2D(filters=nf*4, kernel_size=(1,1), padding="same", use_bias=False)(l1)
		l1 = batchn(bn, l1)

		if i==0:
			l0 = Conv2D(filters=nf*4, kernel_size=(1,1), strides=((2,2) if downsample else (1,1)), padding="same", use_bias=False)(l0)

		l0 = Add()([l1,l0])
		l0 = ReLU()(l0)
	return l0


inp = Input(shape=(32,32,3))
l = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", use_bias=False)(inp)
l = MaxPool2D(pool_size=(2,2), strides=(2,2))(l)
l = resnet_block(l, 64, bn, 3, False)
l = resnet_block(l, 128, bn, 4)
l = resnet_block(l, 256, bn, 6)
l = resnet_block(l, 512, bn, 3)
l = GlobalAveragePooling2D()(l)
out = Dense(units=10, activation="softmax")(l)

model = Model(inp, out)

opt = Adam(lr=0.0001, epsilon=1e-06)
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
	batch_size=100,
	#validation_data=(x_test, y_test)
	)

print("Mean Time:", np.mean(time_callback.times))
if gpu:
	with open("results/keras/keras_resnet50_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
		pickle.dump(hist.history["accuracy"], f)
	with open("results/keras/keras_val_resnet50_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
		pickle.dump(hist.history["val_accuracy"], f)
	print(hist.history["accuracy"], hist.history["val_accuracy"])
