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

#VGG16 sin batchnorm
with open("results/keras/keras_resnet18_no_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_resnet18_no_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_resnet18_no_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_resnet18_no_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_resnet18.txt")
print("Resnet18:", mtime)
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
plt.title("resnet18 sin batchnorm")
plt.legend(["results/keras/keras_tr", "results/keras/keras_te", "results/eddl/eddl_tr", "results/eddl/eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/resnet18_nobn.png")
#plt.show()
plt.clf()


#VGG16 con batchnorm
with open("results/keras/keras_resnet18_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_resnet18_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_resnet18_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_resnet18_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_resnet18_bn.txt")
print("Resnet18 BN:", mtime)
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
plt.title("resnet18 con batchnorm")
plt.legend(["results/keras/keras_tr", "results/keras/keras_te", "results/eddl/eddl_tr", "results/eddl/eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/resnet18_bn.png")
#plt.show()
plt.clf()

#VGG19 sin batchnorm
with open("results/keras/keras_resnet34_no_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_resnet34_no_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_resnet34_no_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_resnet34_no_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_resnet34.txt")
print("Resnet34:", mtime)
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
plt.title("resnet34 sin batchnorm")
plt.legend(["results/keras/keras_tr", "results/keras/keras_te", "results/eddl/eddl_tr", "results/eddl/eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/resnet34_nobn.png")
#plt.show()
plt.clf()

#VGG19 con batchnorm
with open("results/keras/keras_resnet34_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_resnet34_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_resnet34_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_resnet34_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_resnet34_bn.txt")
print("Resnet34 BN:", mtime)
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
plt.title("resnet34 con batchnorm")
plt.legend(["results/keras/keras_tr", "results/keras/keras_te", "results/eddl/eddl_tr", "results/eddl/eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/resnet34_bn.png")
#plt.show()
plt.clf()

#VGG19 sin batchnorm
with open("results/keras/keras_resnet50_no_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_resnet50_no_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_resnet50_no_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_resnet50_no_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_resnet50.txt")
print("Resnet50:", mtime)
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
plt.title("resnet50 sin batchnorm")
plt.legend(["results/keras/keras_tr", "results/keras/keras_te", "results/eddl/eddl_tr", "results/eddl/eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/resnet50_nobn.png")
#plt.show()
plt.clf()

#VGG19 con batchnorm
with open("results/keras/keras_resnet50_batchnorm", "rb") as f:
	acc_keras_tr = pickle.load(f)
with open("results/keras/keras_val_resnet50_batchnorm", "rb") as f:
	acc_keras_te = pickle.load(f)

with open("results/pytorch/pytorch_resnet50_batchnorm", "rb") as f:
	acc_pytorch_tr = pickle.load(f)
with open("results/pytorch/pytorch_val_resnet50_batchnorm", "rb") as f:
	acc_pytorch_te = pickle.load(f)

acc_eddl_tr, acc_eddl_te, mtime = get_eddl_res("results/eddl/eddl_resnet50_bn.txt")
print("Resnet50 BN:", mtime)
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
plt.title("resnet50 con batchnorm")
plt.legend(["results/keras/keras_tr", "results/keras/keras_te", "results/eddl/eddl_tr", "results/eddl/eddl_te", "pytorch_tr", "pytorch_te"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.savefig("results/resnet50_bn.png")
#plt.show()
plt.clf()