import torch

# class_num = 10
# batch_size = 4
# label = torch.LongTensor(batch_size, 1).random_() % class_num

# x = torch.Tensor([2])
#
# w = torch.randn(1, requires_grad=True)
# b = torch.randn(1, requires_grad=True)
#
# y = torch.mul(w, x)
# z = torch.add(y, b)
# print(x.requires_grad, w.requires_grad, b.requires_grad, y.requires_grad, z.requires_grad)
# print(x.is_leaf, w.is_leaf, b.is_leaf, y.is_leaf, z.is_leaf)
# print(x.grad_fn, w.grad_fn, b.grad_fn, y.grad_fn, z.grad_fn)
#
# z.backward()
# print(x.grad, w.grad, b.grad, y.grad, z.grad)
#
# x = torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)
# J = torch.zeros(2, 2)
#
# y = torch.zeros(1, 2)
#
# y[0, 0] = x[0, 0] ** 2 + 3 * x[0, 1]
# y[0, 1] = x[0, 1] ** 2 + 2 * x[0, 0]
#
# # y.backward(torch.Tensor([[1, 1]]))
# y.backward(torch.FloatTensor([1.0, 1.0]))
# print(x.grad)
# y.backward(torch.Tensor([[1, 0]]), retain_graph=True)
# print(x.grad)
# J[0] = x.grad
# y.backward(torch.Tensor([[0, 1]]))
# print(x.grad)
# J[1] = x.grad
# print(J)

# import torch
# x = torch.tensor(1.0, requires_grad=True)
# z = x ** 3
# z.backward()
# print(x.grad.data)
#
# x = torch.tensor([0.0, 2.0, 8.0], requires_grad=True)
# y = torch.tensor([5.0, 1.0, 7.0], requires_grad=True)
#
# z = x * y
# z.backward(torch.FloatTensor([1.0, 1.0, 1.0]))
# print(x.grad.data)
# print(y.grad.data)

# import numpy as np
# import matplotlib.pyplot as plt
#
# np.random.seed(100)
# x = np.linspace(-1, 1, 100).reshape(100, 1)
#
# y = 3 * np.power(x, 2) + 2 + np.random.rand(x.size).reshape(100, 1)
# plt.scatter(x, y)
# plt.show()
#
# w1 = np.random.rand(1, 1)
# b1 = np.random.rand(1, 1)
#
# lr = 0.001
#
# for i in range(10000):
#     y_pred = np.power(x, 2) * w1 + b1
#     loss = 0.5 * (y_pred - y) ** 2
#     loss = loss.sum()
#
#     grad_w = np.sum((y_pred-y) * np.power(x, 2))
#     grad_b = np.sum((y_pred-y))
#
#     w1 -= lr * grad_w
#     b1 -= lr * grad_b
#
# plt.plot(x, y_pred, 'r-', label='predict')
# plt.scatter(x, y, color='blue', marker='o', label='true')
# plt.xlim(-1, 1)
# plt.ylim(2, 6)
# plt.legend()
# plt.show()
# print(w1, b1)

# import torch
# import matplotlib.pyplot as plt
#
# torch.manual_seed(100)
# dtype = torch.float
#
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = 3 * x.pow(2) + 0.2 * torch.rand(x.size())
#
# plt.scatter(x.numpy(), y.numpy())
# plt.show()
#
# w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
# b = torch.randn(1, 1, dtype=dtype, requires_grad=True)
#
# lr = 0.001
# for ii in range(800):
#     y_pred = x.pow(2).mm(2) + b
#     loss = 0.5 * (y_pred-y) ** 2
#     loss = loss.sum()
#
#     loss.backward()
#
#     with torch.no_grad():
#         w -= lr * w.grad
#         b -= lr * b.grad
#
#         w.grad.zero_()  ###注意这里是ｗ和ｂ的梯度清零，而不是ｗ和ｂ的值清零，这里的理解尤其重要．
#         b.grad.zero_()

import numpy as np
import torch

from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epochs = 20
lr = 0.01
momentum = 0.5

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('./data', train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

import matplotlib.pyplot as plt

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title('Ground Truth: {}'.format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
    plt.show()


import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epochs = 20
lr = 0.01

momentum = 0.5

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
# test_dataset = mnist.MNIST('./data', train=False, transform=transform, download=False)
#
# train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
model = Net(28*28, 300, 100, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()

    if epoch % 5 in train_loader:
        optimizer.param_groups[0]['lr'] *= 0.1
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size[0], -1)

        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred==label).sum().item()








