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
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

bn = int(sys.argv[1]) == 1

def batchn(bn, lc):
	if bn:
		return BatchNormalization()(lc)
	else:
		return lc

def resnet_block(l0, nf, bn, downsample=True):
	for i in range(2):
		l1 = Conv2D(filters=nf, kernel_size=(3,3), strides=((2,2) if (downsample and i==0) else (1,1)), padding="same", use_bias=False)(l0)
		l1 = batchn(bn, l1)
		l1 = ReLU()(l1)
		l1 = Conv2D(filters=nf, kernel_size=(3,3), padding="same", use_bias=False)(l1)
		l1 = batchn(bn, l1)

		if i==0 and downsample:
			l0 = Conv2D(filters=nf, kernel_size=(1,1), strides=(2,2), padding="same", use_bias=False)(l0)

		l0 = Add()([l1,l0])
		l0 = ReLU()(l0)
	return l0


inp = Input(shape=(32,32,3))
l = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", use_bias=False)(inp)
l = MaxPool2D(pool_size=(2,2), strides=(2,2))(l)
l = resnet_block(l, 64, bn, False)
l = resnet_block(l, 128, bn)
l = resnet_block(l, 256, bn)
l = resnet_block(l, 512, bn)
l = GlobalAveragePooling2D()(l)
out = Dense(units=10, activation="softmax")(l)

model = Model(inp, out)

from keras.optimizers import Adam
opt = Adam(lr=0.0001, epsilon=1e-06)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()
plot_model(model, to_file="resnet18"+("batchnorm" if bn else "no_batchnorm")+".png", show_shapes=True)

hist = model.fit(x_train, y_train,
	epochs=50,
	batch_size=50,
	validation_data=(x_test, y_test)
	)

with open("results/keras/keras_resnet18_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
	pickle.dump(hist.history["accuracy"], f)
with open("results/keras/keras_val_resnet18_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
	pickle.dump(hist.history["val_accuracy"], f)
print(hist.history["accuracy"], hist.history["val_accuracy"])