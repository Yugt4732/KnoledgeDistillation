import torch
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
import os
from plt import *
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as dst
from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from network import define_tsnet

def model_eval(model, testloader):
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (img, target) in enumerate(test_loader, start=1):
        img = img.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        _, _, _, _, _, out = model(img)
        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
    print('Prec@1: {:.2f}, Prec@5: {:.2f}'.format(top1.avg, top5.avg))


if __name__ == "__main__":



    dataset = dst.CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    test_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_loader = torch.utils.data.DataLoader(
        dataset(root="./datasets",
                transform=test_transform,
                train=False,
                download=False),
        batch_size=128, shuffle=True, pin_memory=True)



#############
    net = define_tsnet("resnet20", 10, 1)
    # save_path = os.path.join('./results/try/initial_r20.pth.tar')
    # torch.save({
    #     'epoch': 0,
    #     'net': net.state_dict(),
    #     'prec@1': 0.0,
    #     'prec@5': 0.0,
    # }, save_path)
    model = torch.load("./results/try/initial_r20.pth.tar")
    load_pretrained_model(net, model['net'])
    net.eval()
    model_eval(net, test_loader)


###############
    tnet = define_tsnet("resnet20", 10, 1)
    checkpoint = torch.load("./results/try/model_best.pth.tar")
    load_pretrained_model(tnet, checkpoint['net'])
    tnet.eval()
    model_eval(tnet, test_loader)

