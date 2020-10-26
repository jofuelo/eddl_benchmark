import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import sys

def resnet_block(l0, nf, bn, reps, downsample):
  for i in range(reps):
    stri = 2 if (downsample and i==0) else 1

    l1 = eddl.GlorotUniform(eddl.Conv(l0, nf, [3,3], [stri,stri], "same", False))
    if(bn):
        l1 = eddl.BatchNormalization(l1, 0.99, 0.001, True, "")
    l1 = eddl.ReLu(l1)
    l1 = eddl.GlorotUniform(eddl.Conv(l1, nf, [3,3], [1,1], "same", False))
    if(bn):
        l1 = eddl.BatchNormalization(l1, 0.99, 0.001, True, "")

    if(stri == 2):
      l0 = eddl.GlorotUniform(eddl.Conv(l0, nf, [1,1], [2,2], "same", False))

    l0 = eddl.Add([l0,l1])
    l0 = eddl.ReLu(l0)
  
  return l0


eddl.download_cifar10()
gpu = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else True

epochs = 50 if gpu else 1
batch_size = 50
num_classes = 10

bn = int(sys.argv[1]) == 1

inp = eddl.Input([3, 32, 32])
l = inp
l = eddl.GlorotUniform(eddl.Conv(l, 64, [7, 7], [2,2], "same", False))
l = eddl.MaxPool(l, [2,2], [2,2], "valid") 
l = resnet_block(l, 64, bn, 2, False)
l = resnet_block(l, 128, bn, 2, True)
l = resnet_block(l, 256, bn, 2, True)
l = resnet_block(l, 512, bn, 2, True)
l = eddl.GlobalAveragePool(l)
l = eddl.Flatten(l)

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
