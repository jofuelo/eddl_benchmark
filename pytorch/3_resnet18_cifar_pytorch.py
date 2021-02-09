import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim, div
from time import time
from torch.utils.data import DataLoader
import sys
import pickle
from torchsummary import summary

bn = int(sys.argv[1]) == 1
gpu = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else True
device = torch.device("cuda:0" if gpu else "cpu")
print("DEVICE:", device)

class ResnetBlock(nn.Module):
    def __init__(self, strides, nf, nf0, reps, bn):
        super(ResnetBlock, self).__init__()
        self.adapt = strides == 2
        self.layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.adapt_layer = nn.Conv2d(nf0, nf, kernel_size=1, stride=strides, padding=0, bias=False) if self.adapt else None
        for i in range(reps):
            if bn:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(nf0, nf, kernel_size=3, stride=strides, padding=1, bias=False),
                    nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                    nn.ReLU(),
                    nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(nf, eps=0.001, momentum=0.99)))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(nf0, nf, kernel_size=3, stride=strides, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False)))
            self.relus.append(nn.ReLU())

            strides = 1
            nf0 = nf

    def forward(self, x):
        for i, (layer, relu) in enumerate(zip(self.layers, self.relus)):
            rama = layer(x)
            if self.adapt and i == 0:
                x = self.adapt_layer(x)
            x = x + rama
            x = relu(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.blocks = nn.Sequential(
            ResnetBlock(1, 64, 64, 2, bn),
            ResnetBlock(2, 128, 64, 2, bn),
            ResnetBlock(2, 256, 128, 2, bn),
            ResnetBlock(2, 512, 256, 2, bn))
        
        self.fcout = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.blocks(out)
        out = out.reshape(out.size(0), -1)
        out = self.fcout(out)
        return out


num_epochs = 50 if gpu else 1
num_classes = 10
batch_size = 50
learning_rate = 0.0001

trans = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root="./dataset_pytorch", train=True, download=True, transform=trans)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset_pytorch", train=False, download=True, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

model = ConvNet()
model.apply(weights_init)


import torch.onnx
from torch.autograd import Variable
dummy_input = Variable(torch.randn(1,3,32,32))
torch.onnx.export(model, dummy_input, "resnet18.onnx")

model=model.to(device)
summary(model, (3,32,32))
print("Cuda version:", torch.version.cuda)
print("Cudnn version:", torch.backends.cudnn.version())
print("Cudnn enabled:", torch.backends.cudnn.enabled)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
acc_list_test = []
times = []
for epoch in range(num_epochs):
    s = time()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    acc_list.append(correct / total)

    print("Train")
    print('Epoch [{}/{}], Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, (correct / total) * 100))

    times.append(time()-s)
    
    if False and gpu:
        total_test = 0
        correct_test = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Run the forward pass
            outputs = model(images)

            # Track the accuracy
            total_test += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()
        acc_list_test.append(correct_test / total_test)
        print("Test")
        print('Epoch [{}/{}], Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, (correct_test / total_test) * 100))
    

print("Mean time:", np.mean(times))
if False and gpu:
    with open("results/pytorch/pytorch_resnet18_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
        pickle.dump(acc_list, f)
    with open("results/pytorch/pytorch_val_resnet18_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
        pickle.dump(acc_list_test, f)
