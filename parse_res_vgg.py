from matplotlib import pyplot as plt
import numpy as np
import pickle

def get_eddl_res(fname):
	with open(fname, "r") as f:
		lines = [l.strip() for l in f.readlines()]
		while lines[0] != "0":
			lines.pop(0)

		times = [float(l[:l.index(" secs/epoch")]) for l in lines if " secs/epoch" in l]

		linestr = [lines[i] for i in range(3,len(lines),8)]
		acc_eddl = [float(l.split("metric[categorical_accuracy]=")[1].split(" ) ")[0]) for l in linestr]

		lineste = [lines[i] for i in range(7,len(lines),8)]
		acc_eddl_te = [float(l.split("metric[categorical_accuracy]=")[1].split(" ) ")[0]) for l in lineste]
		return acc_eddl, acc_eddl_te, round(np.mean(times))

#VGG16 without batchnorm
with open("results/keras/keras_vgg16_no_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_vgg16_no_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_vgg16_no_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_vgg16_no_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_vgg16.txt")
print("VGG16:", mtime)
print("Accuracy")
print("\t- Keras:", round(acc_keras_tr[-1]*100, 1), round(acc_keras_te[-1]*100, 1))
print("\t- Pytorch:", round(acc_pytorch_tr[-1]*100, 1), round(acc_pytorch_te[-1]*100, 1))
print("\t- EDDL:", round(acc_eddl_tr[-1]*100, 1), round(acc_eddl_te[-1]*100, 1))
print()


plt.plot(acc_keras_tr)
plt.plot(acc_keras_te)
plt.plot(acc_eddl_tr)
plt.plot(acc_eddl_te)
plt.plot(acc_pytorch_tr)
plt.plot(acc_pytorch_te)
plt.title("VGG16 without batchnorm")
plt.legend(["keras_tr", "keras_te", "eddl_tr", "eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/vgg16_nobn.png")
#plt.show()
plt.clf()


#VGG16 with batchnorm
with open("results/keras/keras_vgg16_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_vgg16_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_vgg16_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_vgg16_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_vgg16_bn.txt")
print("VGG16 BN:", mtime)
print("Accuracy")
print("\t- Keras:", round(acc_keras_tr[-1]*100, 1), round(acc_keras_te[-1]*100, 1))
print("\t- Pytorch:", round(acc_pytorch_tr[-1]*100, 1), round(acc_pytorch_te[-1]*100, 1))
print("\t- EDDL:", round(acc_eddl_tr[-1]*100, 1), round(acc_eddl_te[-1]*100, 1))
print()

plt.plot(acc_keras_tr)
plt.plot(acc_keras_te)
plt.plot(acc_eddl_tr)
plt.plot(acc_eddl_te)
plt.plot(acc_pytorch_tr)
plt.plot(acc_pytorch_te)
plt.title("VGG16 with batchnorm")
plt.legend(["keras_tr", "keras_te", "eddl_tr", "eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/vgg16_bn.png")
#plt.show()
plt.clf()

#VGG19 without batchnorm
with open("results/keras/keras_vgg19_no_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_vgg19_no_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_vgg19_no_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_vgg19_no_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_vgg19.txt")
print("VGG19:", mtime)
print("Accuracy")
print("\t- Keras:", round(acc_keras_tr[-1]*100, 1), round(acc_keras_te[-1]*100, 1))
print("\t- Pytorch:", round(acc_pytorch_tr[-1]*100, 1), round(acc_pytorch_te[-1]*100, 1))
print("\t- EDDL:", round(acc_eddl_tr[-1]*100, 1), round(acc_eddl_te[-1]*100, 1))
print()


plt.plot(acc_keras_tr)
plt.plot(acc_keras_te)
plt.plot(acc_eddl_tr)
plt.plot(acc_eddl_te)
plt.plot(acc_pytorch_tr)
plt.plot(acc_pytorch_te)
plt.title("VGG19 without batchnorm")
plt.legend(["keras_tr", "keras_te", "eddl_tr", "eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/vgg19_nobn.png")
#plt.show()
plt.clf()

#VGG19 with batchnorm
with open("results/keras/keras_vgg19_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_vgg19_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_vgg19_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_vgg19_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_vgg19_bn.txt")

plt.plot(acc_keras_tr)
plt.plot(acc_keras_te)
plt.plot(acc_eddl_tr)
plt.plot(acc_eddl_te)
plt.plot(acc_pytorch_tr)
plt.plot(acc_pytorch_te)
plt.title("VGG19 with batchnorm")
plt.legend(["keras_tr", "keras_te", "eddl_tr", "eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/vgg19_bn.png")
#plt.show()
plt.clf()

print("VGG19 BN:", mtime)
print("Accuracy")
print("\t- Keras:", round(acc_keras_tr[-1]*100, 1), round(acc_keras_te[-1]*100, 1))
print("\t- Pytorch:", round(acc_pytorch_tr[-1]*100, 1), round(acc_pytorch_te[-1]*100, 1))
print("\t- EDDL:", round(acc_eddl_tr[-1]*100, 1), round(acc_eddl_te[-1]*100, 1))
print()