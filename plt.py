import torch

import matplotlib.pyplot as plt
from utils import AverageMeter, accuracy, transform_time



def curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel("step")
    plt.ylabel("value")
    plt.show()

# def test(test_loader, net, criterion):
# 	losses = AverageMeter()
# 	top1   = AverageMeter()
# 	top5   = AverageMeter()
#
# 	net.eval()
#
# 	end = time.time()
# 	for i, (img, target) in enumerate(test_loader, start=1):
# 		if args.cuda:
# 			img = img.cuda(non_blocking=True)
# 			target = target.cuda(non_blocking=True)
#
# 		with torch.no_grad():
# 			_, _, _, _, _, out = net(img)
# 			loss = criterion(out, target)
#
# 		prec1, prec5 = accuracy(out, target, topk=(1,5))
# 		losses.update(loss.item(), img.size(0))
# 		top1.update(prec1.item(), img.size(0))
# 		top5.update(prec5.item(), img.size(0))
#
# 	f_l = [losses.avg, top1.avg, top5.avg]
# 	logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))
# 	global data1
# 	data1.append(top1.avg)
#
# 	return top1.avg, top5.avg

def ev(model, test_loader):
    model = torch.load(path)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    # test(test_loader, model, criterion)
    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            _, _, _, _, _, out = net(img)
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))


    return top1.avg, top5.avg




def p_img(img, label, name):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081 +0.1307, camp="gtay", interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# def main():


if __name__ == '__main__':
    pass

