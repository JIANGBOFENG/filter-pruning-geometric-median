from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
import models
import numpy as np
import pickle
from scipy.spatial import distance
import pdb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str,default="./datasetpath", help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./svae_for_ck', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate_norm', type=float, default=0.9, help='the remaining ratio of pruning based on Norm')#
parser.add_argument('--rate_dist', type=float, default=0.1, help='the reducing ratio of pruning based on Distance')

parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Norm Pruning Rate: {}".format(args.rate_norm), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_dist), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Dist type: {}".format(args.dist_type), log)

    print_log("this is gpu index:{}".format(torch.cuda.current_device()),log)
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    #pdb.set_trace()
    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    if args.use_pretrain:
        if os.path.isfile(args.pretrain_path):
            print_log("=> loading pretrain model '{}'".format(args.pretrain_path), log)
        else:
            dir = './pretrain_model/cifar10/baseline_without_pruning/'
            # dir = '/data/uts521/yang/progress/cifar10_base/'
            whole_path = dir + 'cifar10_' + args.arch + '_baseline'+'/cifar10_'+ args.arch+'_base1'
            args.pretrain_path = whole_path + '/checkpoint.pth.tar'
            print_log("Pretrain path: {}".format(args.pretrain_path), log)
        pretrain = torch.load(args.pretrain_path)
        if args.use_state_dict:
            net.load_state_dict(pretrain['state_dict'])
        else:
            net = pretrain['state_dict']

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']

            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    m = Mask(net)#创建一个mask
    m.init_length()#得到该模型所有层的参数信息
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    #剩余部分的比率，从这个地方看，应该是先按照norm，保留rate_norm比例的参数，然后按照dist剪掉rate_dist部分
    print("total remaining ratio is %f" % (args.rate_norm - args.rate_dist))
    
    val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)

    print(" accu before is: %.3f %%" % val_acc_1)

    m.model = net

    m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)#初始化mask，获取codebook
    #    m.if_zero()
    m.do_mask()
    m.do_similar_mask()
    net = m.model#加载按照mask更新权重后的网络
    #    m.if_zero()
    if args.use_cuda:
        net = net.cuda()
    val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)#测试一下看看
    print(" accu after is: %s %%" % val_acc_2)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    small_filter_index = []
    large_filter_index = []

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),#目前为止最大的val__acc
                                                               100 - recorder.max_accuracy(False)), log)#目前为止最小的error

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, m)

        # evaluate on validation set
        val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)
        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:#如果到需要剪枝的epoch，则进行剪枝
            m.model = net#把模型导入mask里
            m.if_zero()#看有多少零
            m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            m.do_mask()
            m.do_similar_mask()
            m.if_zero()#看看零还在不在了
            net = m.model#把剪枝后的模型返回给net
        #注意：这种剪枝方法是软剪枝，可以不断恢复的，并不是剪掉之后就不用了。    
            if args.use_cuda:
                net = net.cuda()

        val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')#保存本次epoch的前向模型

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    log.close()


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, m):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        m.do_grad_mask()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        print("\n"+"______________________________"+"This for debug")
        loss = criterion(output, target_var)
        print(loss)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        print("\n"+"______________________________"+"This for debug")
        print(prec1)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                   error1=100 - top1.avg),
              log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()#刷新缓冲区


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')#如果是最好的，则保存为best
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):#采用LR decay方法调整学习率，在160之前是0.1，在150-225是0.01，在之后是0.001。
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    #Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {} 
        self.mat = {}#这个矩阵是用来存放norm的mask
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}#用来存放distance的mask
        self.norm_matrix = {}#没用过，不知道这个是干啥的

    def get_codebook(self, weight_torch, compress_rate, length):#这个函数没有用过，不知道干啥
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)#对weight进行排序，返回一个升序序列

        threshold = weight_sort[int(length * (1 - compress_rate))]#找到最小的这些权重，然后定义第剪枝率*全部个数的值是阈值
        weight_np[weight_np <= -threshold] = 1#好像写得不太对？按理来说应该是压缩了除以2，咋也不知道为啥
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):#赋值给self.mat
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))#被剪枝掉的个数
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]#返回按照范数从小到大排列，最小的filter_pruned_num个通道的索引
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)#产生一个和本层权重个数相同的字典
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))#需要按照norm剪掉的通道数量
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)#需要根据distance剪掉的通道数量
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)#按通道数区分开，顺序排布为w向量

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)#按通道，返回weight_vec的L2范数，dim=1即返回每个通道的L2范数，即squrt(sum(x^2))
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
#             ">> a = torch.randn(4, 4)
#              >>> a
#           tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
#                   [ 0.1598,  0.0788, -0.0745, -1.2700],
#                   [ 1.2208,  1.0722, -0.7064,  1.2564],
#                   [ 0.0669, -0.2318, -0.8229, -0.9280]])


#              >>> torch.argsort(a, dim=1)
#           tensor([[2, 0, 3, 1],
#                   [3, 2, 1, 0],
#                   [2, 1, 0, 3],
#                   [3, 2, 1, 0]])"
#           np.argsort返回矩阵，该矩阵沿着给定维度进行排序，返回是一个在给定维度下升序序列的原索引，filter_pruned_num:表示从第N个起到最后一个,此处返回的L2范数较大的索引值。
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]#：filter_pruned_num表示从第0个到该数索引N-1,即较小的通道索引

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()#获取norm较大通道的索引
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()#index_select按照indices在第0轴（行）进行抽取，即按照索引抽出了L2范数较大的通道向量
            # for euclidean distance
            if dist_type == "l2" or "l1":#计算weight_vec_after_norm（较大范数的通道各个w）之间的欧氏距离，返回一个表示距离的矩阵，
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')#相似矩阵
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)#二维矩阵，行相加，

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]#按照通道剪枝的数量排布，距离和最大的索引组
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]
#           similar_index_for_filter 是一个选出来和其他filter具有较小范数距离和的filter索引组，一共有similar_pruned_num个。
            print('filter_large_index', filter_large_index)
            print('filter_small_index', filter_small_index)
            print('similar_sum', similar_sum)
            print('similar_large_index', similar_large_index)
            print('similar_small_index', similar_small_index)
            print('similar_index_for_filter', similar_index_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]#每个kernel内部的权重数量
            for x in range(0, len(similar_index_for_filter)):
                #按照选出来具有较小范数和的通道索引组的索引，把这个字典中相应索引位置的（一个kernel内部权重的个数）变为0
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        """保存 mdoel_length 数组,其中存折每一层的size大小"""
        for index, item in enumerate(self.model.parameters()):#model.parameters()是一个迭代器，每次返回的是Weights和Bais参数的值
            self.model_size[index] = item.size()#model_size顺序存储了每一层的大小

        for index1 in self.model_size:#此举目的是计算每一层的大小size，存在model_length里
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        for index, item in enumerate(self.model.parameters()):#按照model的结构，生成一个初始rate全为1，长度和model层数相同的数组compress_rate，distance_rate[index]
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):#把剪枝率赋进去，第三个是步长，range一般左闭右开
            self.compress_rate[key] = rate_norm_per_layer
            self.distance_rate[key] = rate_dist_per_layer
        # different setting for  different architecture
        if args.arch == 'resnet20':
            last_index = 57
        elif args.arch == 'resnet32':
            last_index = 93
        elif args.arch == 'resnet56':
            last_index = 165
        elif args.arch == 'resnet110':
            last_index = 327
        # to jump the last fc layer
        self.mask_index = [x for x in range(0, last_index, 3)]#按照不同结构，生成mask的索引

    #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
            # mask for norm criterion#为范数正则项制定codebook作为mask
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()

                # # get result about filter index
                # self.filter_small_index[index], self.filter_large_index[index] = \
                #     self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

            # mask for distance criterion#为距离正则项制定codebook作为mask
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])#拉成一行，一维的
                b = a * self.mat[index]#按照较小的norm的channel的索引，赋值为0
                item.data = b.view(self.model_size[index])#重新拉成model大小
        print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])#拉成一行，一维的
                b = a * self.similar_matrix[index]#按照较小的norm的channel的索引，赋值为0
                item.data = b.view(self.model_size[index])#重新拉成model大小
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])#取出参数梯度，排列为1列
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                b = a * self.mat[index]#按照mask取参数梯度
                b = b * self.similar_matrix[index]#按照distance取参数梯度
                item.grad.data = b.view(self.model_size[index])#重新排列后放回去
        # print("grad zero Done")

    def if_zero(self):
        """此操作是用来打印权重系数中非零的个数，剪枝前后打印一次，证明剪掉了零的权重"""
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
