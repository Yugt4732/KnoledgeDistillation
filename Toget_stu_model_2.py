import torch
import torch
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
# device = 'cpu'

train_loader = datasets.MNIST('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
                       ]))

# print(type(train_loader))


print("-"*10)
trainx = []
trainy = []
num_images = 100
for class_label in range(5):
    idx = 0
    for i,label in enumerate(train_loader):
        if label[1] == class_label:
            trainx.append(train_loader[i][0].numpy())
            trainy.append(label[1])
            idx += 1
        if idx == num_images:
            break

class LeNet32(nn.Module):
    def __init__(self):
        super(LeNet32, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)            #  w=(in_w-fil_w+2*pd)/sd+1
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同    # w=(28-5+4)/1+1 =28
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),           #w=(14-5)/1+1 = 10, h=10
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 5)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # F.softmax(x, dim=1)

def train(epoch, model, device, train_loader,optimizer, interval):
    losses = []
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader,leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses, correct / len(train_loader.dataset)



model = LeNet32()




trainx = np.asarray(trainx)
trainy = np.asarray(trainy)
trainx = torch.FloatTensor(trainx)
trainy = torch.LongTensor(trainy)

rand_perm = torch.randperm(len(trainx))
trainx = trainx[rand_perm]
trainy = trainy[rand_perm]

optimizer = optim.Adam(model.parameters())
model.train()
batch_size = 64
for epoch in range(1,11):
    for i in range(0, len(trainx), batch_size):
        data = trainx[i:i + batch_size]
        target = trainy[i:i + batch_size]
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    accu = (model(trainx).argmax(dim=1) == trainy).sum().item() / len(trainx)
    print("Accu = {:.4f}".format(accu))

###################################for test_1
model.eval()
test_loader = datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
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

accu = (model(testx).argmax(dim=1) == testy).sum().item() / len(testx)
print("%4f" % accu)

##############################for_test_2

test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
    datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ])),
    batch_size=64, shuffle=True)
model.eval()  # 设置为test模式
test_loss = 0  # 初始化测试损失值为0
correct = 0  # 初始化预测正确的数据个数为0
# print(len(test_loader.dataset))
for data, target in test_loader:
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
accu = correct/len(test_loader)
print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
   correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

save_root = './results'
state = {
            'epoch': epoch,
            'net': model.state_dict(),
            'prec@1': accu
        }
save_path = os.path.join(save_root, 'model_stu.pth.tar')
torch.save(state, save_path)
torch.save(model.state_dict(), 'stu_model.pth.tar')  # 保存模型
# print(model)
