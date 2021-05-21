import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# print('Done')


import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)  ### 5 * 5 * 3  * 16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1) ### 3 * 3 * 16 * 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1296, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36*6*6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x

net = CNNNet()
net = net.to(device)
# for x in net.parameters():
#     print(x.numel())


print('net have {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

import torch.optim as optim
LR = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(list(net.children()))

print(net)
print(nn.Sequential(*list(net.children())[:4]))

for epoch in range(10):
    print(epoch)
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # labels.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch +1, i+1, running_loss / 2000))
print('Finished Training')

# dataiter = iter(testloader)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# images, labels = images.cuda(), labels.cuda()
#
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(4)))


# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.cuda(), labels.cuda()
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted==labels).sum().item()
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct/total))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.aap(x)
        x = x.view(x.shape[0], -1)
        return x
#
# net = Net()
# net = net.cuda()

# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader):
#         inputs, labels = data
#         inputs, labels = inputs.cuda(), labels.cuda()
#
#         outputs = net(inputs)
#
#         optimizer.zero_grad()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         if i% 200 == 199:
#             print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/200))
#             # running_loss = 0.0

