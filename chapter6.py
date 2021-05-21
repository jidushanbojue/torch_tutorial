# import torch.nn as nn
# import torch.nn.functional as F
# import torch
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# class CNNNet(nn.Module):
#     def __init__(self):
#         super(CNNNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=5, stride=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(1296, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = x.view(-1, 36*6*6)
#         x = F.relu(self.fc2(F.relu(self.fc1(x))))
#
# # net = CNNNet()
# # net = net.to(device)
# # print('Done')
# #
# # m1 = nn.MaxPool2d(3, stride=2)
# # m2 = nn.MaxPool2d((3, 2), stride=(2, 1))
# #
# # input = torch.randn(20, 15, 50, 32)
# # output = m2(input)
# # print(output.shape)
#
# # m = nn.AdaptiveAvgPool2d((5, 7))
# # input = torch.randn(1, 64, 8, 9)
# # output = m(input)
#
# # m = nn.AdaptiveMaxPool2d(7)
# # input = torch.randn(1, 64, 10, 9)
# # output = m(input)
#
# m = nn.AdaptiveMaxPool2d((None, 7))
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
#
#
# print('Done')



