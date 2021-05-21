# import torch
# from torch import nn
# import torch.nn.functional as F

# m = nn.Sigmoid()
# input = torch.randn(2)
# output = m(input)
# print('Done')
#
# torch.nn.MSELoss()

# torch.manual_seed(10)
# 
# loss = nn.MSELoss(reduction='mean')
# input = torch.randn(1, 2, requires_grad=True)
# print(input)
# 
# target = torch.randn(1, 2)
# print(target)


# torch.manual_seed(10)
# loss = nn.CrossEntropyLoss()
#
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long)
# target = target.random_(5)
#
# output = loss(input, target)
# output.backward()
# print('Done')

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
boston = load_boston()

X, y = (boston.data, boston.target)
dim = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
num_train = X_train.shape[0]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train -= mean
X_train /= std

X_test -= mean
X_test /= std

train_data = torch.from_numpy(X_train)

dtype = torch.FloatTensor
train_data.type(dtype)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = torch.from_numpy(X_train).float()
train_target = torch.from_numpy(y_train).float()

test_data = torch.from_numpy(X_test).float()
test_target = torch.from_numpy(y_test).float()

####　这种写法，居然不需要定义model，而是直接定义，这样也是可以的，一定得灵活一点，
####　因为定义class的话，我们还是需要初始化的，这样的化，我们就不需要初始化．

net1_overfitting = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

net2_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.BatchNorm1d(num_features=16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.BatchNorm1d(num_features=32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

net1_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 8),
    torch.nn.BatchNorm1d(num_features=8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4),
    torch.nn.BatchNorm1d(num_features=4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
)

net1_dropped = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

loss_func = torch.nn.MSELoss()

optimizer_ofit = torch.optim.Adam(net1_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net1_dropped.parameters(), lr=0.01)
optimizer_nb = torch.optim.Adam(net1_nb.parameters(), lr=0.01)

from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs')

for epoch in range(200):
    print(epoch)
    net1_overfitting.train()
    net1_dropped.train()
    net1_nb.train()

    pred_ofit = net1_overfitting(train_data)
    pred_drop = net1_dropped(train_data)
    pred_nb = net1_nb(train_data)

    loss_ofit = loss_func(pred_ofit, train_target)
    loss_drop = loss_func(pred_drop, train_target)
    loss_nb = loss_func(pred_nb, train_target)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    optimizer_nb.zero_grad()

    loss_ofit.backward()
    loss_drop.backward()
    loss_nb.backward()

    optimizer_ofit.step()
    optimizer_drop.step()
    optimizer_nb.step()

    writer.add_scalars('train_group_loss', {'trainloss_ofit': loss_ofit.item(), 'trainloss_drop': loss_drop.item(), 'trainloss_nb': loss_nb.item()}, epoch)

    net1_overfitting.eval()
    net1_dropped.eval()
    net1_nb.eval()

    test_pred_orig = net1_overfitting(test_data)
    test_pred_drop = net1_dropped(test_data)
    test_pred_nb = net1_nb(test_data)

    orig_loss = loss_func(test_pred_orig, test_target)
    drop_loss = loss_func(test_pred_drop, test_target)
    nb_loss = loss_func(test_pred_nb, test_target)

    writer.add_scalars('test_group_loss', {'testloss_ofit': orig_loss.item(), 'testloss_drop': drop_loss.item(), 'testloss_nb': nb_loss.item()}, epoch)







