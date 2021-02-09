import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import sys

def defblock(l, bn, nf, reps):
    for i in range(reps):
      l = eddl.GlorotUniform(eddl.Conv(l, nf, [3, 3]))
      if bn: 
        l = eddl.BatchNormalization(l, 0.99, 0.001, True, "")
      l = eddl.ReLu(l)
    l = eddl.MaxPool(l, [2, 2], [2, 2], "valid")
    return l


eddl.download_cifar10()
gpu = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else True

epochs = 10 if gpu else 1
batch_size = 100
num_classes = 10

bn = int(sys.argv[1]) == 1


inp = eddl.Input([3, 32, 32])
l = inp
l = defblock(l, bn, 64, 2)
l = defblock(l, bn, 128, 2)
l = defblock(l, bn, 256, 3)
l = defblock(l, bn, 512, 3)
l = defblock(l, bn, 512, 3)
l = eddl.Flatten(l)
for i in range(2):
    l = eddl.GlorotUniform(eddl.Dense(l, 4096))
    if(bn):
        l = eddl.BatchNormalization(l, 0.99, 0.001, True, "")
    l = eddl.ReLu(l)

out = eddl.Softmax(eddl.GlorotUniform(eddl.Dense(l, num_classes)))

net = eddl.Model([inp], [out])
eddl.plot(net, "model.pdf")

eddl.build(net,
    eddl.adam(0.0001),
    ["soft_cross_entropy"],
    ["categorical_accuracy"],
    eddl.CS_GPU() if gpu else eddl.CS_CPU()
)

eddl.summary(net)

x_train = Tensor.load("cifar_trX.bin")
y_train = Tensor.load("cifar_trY.bin")
x_train.div_(255)

x_test = Tensor.load("cifar_tsX.bin")
y_test = Tensor.load("cifar_tsY.bin")
x_test.div_(255)

from time import time
import numpy as np
tiempos = []
for i in range(epochs):
    s = time()
    print(i)
    res = eddl.fit(net,[x_train],[y_train],batch_size, 1)
    tiempos.append(time()-s)
    if gpu:
        print("Evaluate test")
        res1 = eddl.evaluate(net,[x_test],[y_test])
print(np.mean(tiempos))
