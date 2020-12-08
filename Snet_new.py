from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import torch.optim as optim
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
from plt import *
import time
import logging
import argparse
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst
from torchvision import datasets, transforms
from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet

parser = argparse.ArgumentParser(description='train student net')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=15, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=0)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, default= 'mnist', help='name of dataset') # cifar10/cifar100
# parser.add_argument('--net_name', type=str, required=True, help='name of basenet')  # resnet20/resnet110
parser.add_argument('--net_name', type=str, default= 'resnet20', help='name of basenet')  # resnet20/resnet110



args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
data1 = []

def main():
    global data1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     cudnn.enabled = True
    #     cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    logging.info('----------- Network Initialization --------------')
    net = define_tsnet(name=args.net_name, num_class=args.num_class)
    logging.info('%s', net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info('-----------------------------------------------')

    # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'model_stu.pth.tar')
    torch.save({
        'epoch': 0,
        'net': net.state_dict(),
        'prec@1': 0.0,
        'prec@5': 0.0,
    }, save_path)

    # initialize optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr = args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay,
                                nesterov = True)

    # define loss functions
    # if args.cuda:
    #     criterion = torch.nn.CrossEntropyLoss().cuda()
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    dataset = dst.MNIST
    train_transform = transforms.Compose([
			transforms.Pad(4, padding_mode='reflect'),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])


    test_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

# define data loader
    train_loader = dataset(root=args.img_root,
                transform=train_transform,
                train=True,
                download=False)

    test_loader = dataset(root=args.img_root,
                transform=test_transform,
                train=False,
                download=True)

    test_loader_2 = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=64, shuffle=True)



###############train_set
    print("-" * 10)
    trainx = []
    trainy = []
    num_images = 65
    for class_label in range(5):
        idx = 0
        for i, label in enumerate(train_loader):
            if label[1] == class_label:
                trainx.append(train_loader[i][0].numpy())
                trainy.append(label[1])
                idx += 1
            if idx == num_images:
                break

    trainx = np.asarray(trainx)
    trainy = np.asarray(trainy)
    trainx = torch.FloatTensor(trainx).cpu()
    trainy = torch.LongTensor(trainy).cpu()

    rand_perm = torch.randperm(len(trainx))
    trainx = trainx[rand_perm]
    trainy = trainy[rand_perm]



#######test_set


    testx = []
    testy = []
    num_images = 65
    for class_label in range(5):
        idx = 0
        for i, label in enumerate(test_loader):
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



    batch_size = 64
    optimizer = optim.Adam(net.parameters())
    net.train()
    for epoch in range(1, args.epochs + 1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        for i in range(0, len(trainx), batch_size):
            data = trainx[i:i + batch_size]
            target = trainy[i:i + batch_size]
            optimizer.zero_grad()
            _, _, _, _, _, output = net(data)
            output = output.cuda()
            target = target.cuda()
            # print(output)
            loss = F.cross_entropy(output, target).cuda()
            loss.backward()
            optimizer.step()
        _, _, _, _, _, out = net(trainx)
        out = out.cuda()
        trainx = trainx.cuda()
        temp = (out.argmax(dim=1) == trainy.cuda()).cuda()
        accu = temp.sum().item() / len(trainx)
        print("accu = {:.4f}".format(accu))



        # evaluate on testing set
        logging.info('Testing the models......')
        # test_top1, test_top5 = test(test_loader, net, criterion)
        # net.eval()
        _, _, _, _, _, out = net(testx)
        out = out.cuda()
        testx = testx.cuda()
        testy = testy.cuda()
        Accu = (out.argmax(dim=1) == testy).sum().item() / len(testx)
        print("Accu= {:.4f}".format(Accu))

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        ###############test in all

        test_loss = 0  # 初始化测试损失值为0
        correct = 0  # 初始化预测正确的数据个数为0
        # print(len(test_loader.dataset))
        for data, target in test_loader_2:
            _, _, _, _, _, output = net(data)
            output = output.cuda()
            target = target.cuda()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum()  # 对预测正确的数据个数进行累加
        accu = correct / len(test_loader_2)
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader_2.dataset),
            100. * correct / len(test_loader_2.dataset)))

        # save model
        is_best = False
        logging.info('Saving models......')
        save_({
            'epoch': epoch,
            'net': net.state_dict(),
            'prec@1': Accu,
        }, is_best, args.save_root)

def save_(state, is_best, save_root):
	save_path = os.path.join(save_root, 'model_stu_point.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_stu.pth.tar')
		shutil.copyfile(save_path, best_save_path)


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

if __name__ == '__main__':
    main()