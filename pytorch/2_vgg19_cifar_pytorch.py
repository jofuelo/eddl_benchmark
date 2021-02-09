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

bn = int(sys.argv[1]) == 1
gpu = int(sys.argv[2]) == 1 if len(sys.argv) > 2 else True
device = torch.device("cuda:0" if gpu else "cpu")
print("DEVICE:", device)

num_epochs = 50 if gpu else 1
num_classes = 10
batch_size = 50
learning_rate = 0.00001

trans = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root="./dataset_pytorch", train=True, download=True, transform=trans)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset_pytorch", train=False, download=True, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def createBlock(nf0, nf, bn, reps):
    if bn:
        if reps == 2:
            return nn.Sequential(
                nn.Conv2d(nf0, nf, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            return nn.Sequential(
                nn.Conv2d(nf0, nf, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nf, eps=0.001, momentum=0.99),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
    else:
        if reps == 2:
            return nn.Sequential(
                nn.Conv2d(nf0, nf, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            return nn.Sequential(
                nn.Conv2d(nf0, nf, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = createBlock(3, 64, bn, 2)
        self.layer2 = createBlock(64, 128, bn, 2)
        self.layer3 = createBlock(128, 256, bn, 4)
        self.layer4 = createBlock(256, 512, bn, 4)
        self.layer5 = createBlock(512, 512, bn, 4)
        if bn:
            self.fc1 = nn.Sequential(
                nn.Linear(512, 4096),
                nn.BatchNorm1d(4096, eps=0.001, momentum=0.99),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096, eps=0.001, momentum=0.99),
                nn.ReLU())
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU())
        self.fcout = nn.Sequential(nn.Linear(4096, 10))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fcout(out)
        return out

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if bn:
            nn.init.xavier_uniform_(m.weight.data)
        else:
            nn.init.kaiming_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

model = ConvNet()
model.apply(weights_init)
model.to(device)

from torchsummary import summary
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
    with open("results/pytorch/pytorch_vgg19_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
        pickle.dump(acc_list, f)
    with open("results/pytorch/pytorch_val_vgg19_"+("batchnorm" if bn else "no_batchnorm"), "wb") as f:
        pickle.dump(acc_list_test, f)
