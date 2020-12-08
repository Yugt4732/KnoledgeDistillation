from network import *
import sys
import logging
from utils import load_pretrained_model, save_checkpoint
import os
import numpy as  np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
MNIST_PATH = './datasets'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join("./results", 'model_stu_test.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('----------- Network Initialization --------------')


model = resnet20(10)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load("./results/model_stu_final.pth.tar")
load_pretrained_model(model, checkpoint['snet'])
model = model.cuda()

model.eval()


test_loader = datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ]))

testx = []
testy = []
num_images = 100
for class_label in range(5, 10):
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


test_loader = datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ]))

testx = []
testy = []
num_images = 100
for class_label in range(5, 10):
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

# ddd = np.concatenate(testx.cpu().numpy(), axis=2).reshape(36,-1)[:,0:360]
# # ddd -= ddd.mean()
# # ddd /= ddd.std()
#
# # plt.subplot(1,2,1)
# plt.imshow(ddd, cmap='gray')
#
# _, _, _, _, _, out = model(testx)
# print("Without mask\t", out.argmax(dim=1)[[i for i in range(1, 10)]].detach().cpu().numpy())
# plt.show()

rand_perm = torch.randperm(len(testx))
testx = testx[rand_perm]
testy = testy[rand_perm]
testx = testx.cuda()
testy = testy.cuda()
_, _, _, _, _, result = model(testx)
result = result.cuda()

cnt = (result.argmax(dim=1) == testy).sum().item()
accu = (result.argmax(dim=1) == testy).sum().item() / len(testx)
logging.info("Accuracy in test_set(5-9): %4f" % accu)
print("cnt= %d" % cnt )

plt.axis('off')

ddd = np.concatenate(testx.cpu().numpy(), axis=2).reshape(36,-1)[:,0:360]
# ddd -= ddd.mean()
# ddd /= ddd.std()

# plt.subplot(1,2,1)
plt.imshow(ddd, cmap='gray')

_, _, _, _, _, out = model(testx)
print("Without mask\t", out.argmax(dim=1)[[i for i in range(1, 10)]].detach().cpu().numpy())
plt.show()

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
for data, target in test_loader:
    data = data.cuda()
    target = target.cuda()
    _, _, _, _, _, output = model(data)
    output = output.cuda()
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).sum()  # 对预测正确的数据个数进行累加
accu = (float)(100.0 * correct / len(test_loader.dataset))
logging.info('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
   correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
