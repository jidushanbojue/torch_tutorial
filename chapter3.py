# import os
# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.get_device_capability(0))
# print(torch.cuda.get_device_properties(0))
#
#
#
# import pandas as pd
# from matplotlib import pyplot as plt
#
# import numpy as np
#
# print(torch.cuda._get_device_index(0))
#
# print(torch.cuda.get_device_name(0))

# import numpy as np
# import torch
from torchvision.datasets import mnist
# from torchvision.transforms import transforms
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.optim as optim
# from torch import nn

# train_batch_size = 64
# test_batch_size = 128
# learning_rate = 0.01
# num_epochs = 20
# lr = 0.01
# #
# momentum = 0.5

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
# test_dataset = mnist.MNIST('./data', train=False, transform=transform, download=False)
#
# train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# class Net(nn.Module):
#     def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
#         super(Net, self).__init__()
#         self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1))
#         self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
#         self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
#
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = self.layer3(x)
#         return x
# #
# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# #
# # model = Net(28*28, 300, 100, 10) ###这里必须先实例化，然后以函数调用的方式调用实例化的对象，并传入数据，
# # model.to(device)
# #
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# #
# # losses = []
# # acces = []
# # eval_losses = []
# # eval_acces = []
# #
# # for epoch in range(num_epochs):
# #     train_loss = 0
# #     train_acc = 0
# #     model.train()
# #     if epoch % 5 == 0:
# #         optimizer.param_groups[0]['lr'] *= 0.1
# #     for img, label in train_loader:
# #         img = img.to(device)
# #         label = label.to(device)
# #         img = img.view(img.size(0), -1)
# #
# #         out = model(img)
# #         loss = criterion(out, label)
# #
# #
# #
# #
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #
# #         train_loss += loss.item()
# #
# #         _, pred = out.max(1)
# #         num_correct = (pred == label).sum().item()
# #         acc = num_correct / img.shape[0]
# #         train_acc += acc
# #
# #     losses.append(train_loss/len(train_loader))
# #     acces.append(train_acc/len(train_loader))
# #
# #     eval_loss = 0
# #     eval_acc = 0
# #     model.eval()
# #
# #     for img, label in test_loader:
# #         img = img.to(device)
# #         label = label.to(device)
# #         img = img.view(img.size(0), -1)
# #         out = model(img)
# #
# #         loss = criterion(out, label)
# #
# #         eval_loss += loss.item()
# #         _, pred = out.max(1)
# #         num_correct = (pred == label).sum().item()
# #         acc = num_correct / img.shape[0]
# #         eval_acc += acc
#
# from collections import OrderedDict
# # class Net2(nn.Module):
# #     def __init__(self):
# #         super(Net2, self).__init__()
# #         self.conv = torch.nn.Sequential(
# #             OrderedDict([
# #                 ('conv1', torch.nn.Conv2d(3, 32, 3, 1, 1)),
# #                 ('relu', torch.nn.ReLU()),
# #                 ('pool', torch.nn.MaxPool2d(2))
# #             ])
# #         )
# #
# #         self.dense = torch.nn.Sequential(
# #             OrderedDict([
# #                 ('dense1', torch.nn.Linear(32*3*3, 128)),
# #                 ('relu2', torch.nn.ReLU()),
# #                 ('dense2', torch.nn.Linear(128, 10))
# #             ])
# #         )
#
# net = Net(28*28, 300, 100, 10)
# # net2 = Net2()
# # print('Done')
#
#
# optimzer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
# print(optimzer.param_groups[1])
# print('done')

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import torch.utils.data.

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net()
net_Momentum = Net()
net_RMSProp = Net()
net_Adam = Net()

nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]

loss_func = torch.nn.MSELoss()
loss_his = [[], [], [], []]

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        print(step)
        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(batch_x)
            loss = loss_func(output, batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            l_his.append(loss.data.numpy())

labels = ['SGD', 'Momentum', 'RMSProp', 'Adam']

for i, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()





