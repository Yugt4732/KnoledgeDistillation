import torch
import torch
from network import *
import sys
import logging
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as  np
import torch.utils.data as Data
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
MNIST_PATH = './datasets'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join("./results/try", 'model_stu_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('----------- Network Initialization --------------')

train_loader = datasets.MNIST('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4, padding_mode='reflect'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
                       ]))

# print(type(train_loader))
model = resnet20(10)

# print("-"*10)
trainx = []
trainy = []
num_images = 129
logging.info("train_EachNum is: %d", num_images)
for class_label in range(5):
    idx = 0
    for i,label in enumerate(train_loader):
        if label[1] == class_label:
            trainx.append(train_loader[i][0].numpy())
            trainy.append(label[1])
            idx += 1
        if idx == num_images:
            break







trainx = np.asarray(trainx)
trainy = np.asarray(trainy)
trainx = torch.FloatTensor(trainx)
trainy = torch.LongTensor(trainy)

rand_perm = torch.randperm(len(trainx))
trainx = trainx[rand_perm]
trainy = trainy[rand_perm]

optimizer = optim.Adam(model.parameters())
model.train()
batch_size = 128
epochs = 10
logging.info("Batch_size is: %d", batch_size)
logging.info("Epochs is: %d", epochs)
for epoch in range(1,epochs+1):
    for i in range(0, len(trainx), batch_size):
        data = trainx[i:i + batch_size]
        target = trainy[i:i + batch_size]
        optimizer.zero_grad()
        # print(data)
        # print(type(data))
        # print(data.shape)
        _, _, _, _, _, output = model(data)
        # print(output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    _, _, _, _, _, result = model(trainx)
    accu = (result.argmax(dim=1) == trainy).sum().item() / len(trainx)
    print("Accu = {:.4f}".format(accu))

###################################for test_1
model.eval()
test_loader = datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ]))

testx = []
testy = []
num_images = 100
for class_label in range(5):
    idx = 0
    for i,label in enumerate(test_loader):
        if label[1] == class_label:
            testx.append(test_loader[i][0].numpy())
            testy.append(label[1])
            idx += 1
        if idx == num_images:
            break

testx = np.asarray(testx)
testy = np.asarray(testy)
testx = torch.FloatTensor(testx)
testy = torch.LongTensor(testy)

rand_perm = torch.randperm(len(testx))
testx = testx[rand_perm]
testy = testy[rand_perm]
_, _, _, _, _, result = model(testx)
accu = (result.argmax(dim=1) == testy).sum().item() / len(testx)
# print("Accuracy in test_set(0-4): %4f" % accu)
logging.info("Accuracy in test_set(0-4): %4f" % accu)

##############################for_test_2

test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
    datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ])),
    batch_size=64, shuffle=True)
model.eval()  # 设置为test模式
test_loss = 0  # 初始化测试损失值为0
correct = 0  # 初始化预测正确的数据个数为0
# print(len(test_loader.dataset))
for data, target in test_loader:
    _, _, _, _, _, output = model(data)
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
accu = (float)(100.0 * correct / len(test_loader.dataset))
# print(accu)
# accu = accu.numpy()
# print(accu)
# print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
#    correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))
logging.info('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
   correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
save_root = './results/try'
state = {
            'epoch': epochs,
            'net': model.state_dict(),
            'prec@1': accu
        }
save_path = os.path.join(save_root, 'model_stu_point.pth.tar')
torch.save(state, save_path)
# torch.save(model.state_dict(), 'stu_model.pth.tar')  # 保存模型
# print(model)
##############################################
test_loader = datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ]))
testx_2 = []
testy_2 = []
num_images = 100
for class_label in range(5, 10):
    idx = 0
    for i,label in enumerate(test_loader):
        if label[1] == class_label:
            testx_2.append(test_loader[i][0].numpy())
            testy_2.append(label[1])
            idx += 1
        if idx == num_images:
            break

testx_2 = np.asarray(testx_2)
testy_2 = np.asarray(testy_2)
testx_2 = torch.FloatTensor(testx_2)
testy_2 = torch.LongTensor(testy_2)

rand_perm = torch.randperm(len(testx_2))
testx_2 = testx_2[rand_perm]
testy_2 = testy_2[rand_perm]
_, _, _, _, _, result = model(testx_2)
accu = (result.argmax(dim=1) == testy_2).sum().item() / len(testx_2)
# print("Accuracy in test_set(0-4): %4f" % accu)
logging.info("Accuracy in test_set(5-9): %4f" % accu)