
import torch

from torchvision import transforms
from torch.utils import data
import utils
import time
import os

from cifar100_split import Cifar100Split

class Trainer(object):
    def __init__(self, start_num=0, end_num=10, net=None,save_path=""):
        self.lr = 0.001
        self.epoch = 50
        self.warm = 1
        self.batch_size = 64
        self.start_num = start_num
        self.end_num = end_num
        self.class_num = end_num - start_num
        self.use_cuda = True
        self.task_num = 1
        self.save_path = save_path
        self.main_net_path = save_path+"/Joint_"+str(end_num)+".ptn"
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = Cifar100Split(start_num=start_num, end_num=end_num, train=True, transform=transform_train)
        testset = Cifar100Split(start_num=start_num, end_num=end_num, train=False, transform=transform_test)
        self.trainloader = data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.testloader = data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=False)
        milestones=[20, 30, 40, 40, 50]
        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.5)
        self.params_set = [self.net.conv1, self.net.conv2_x, self.net.conv3_x, self.net.conv4_x, self.net.conv5_x]
        self.optmizer_De_list = []
        self.scheduler_De_list = []
        self.decoder_list = []
        self.pixelwise_loss = torch.nn.L1Loss()
        self.best_loss_pix = {}


    def test_acc(self,data_loader,start_num,end_num):
        self.net.eval()
        # test_loss = 0.0  # cost function error
        correct_top1 = []
        correct_top5 = []
        for (inputs, labels) in data_loader:

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)

                prec1, prec5 = self.accuracy(outputs.data[:, start_num:end_num],
                                             labels.cuda().data, topk=(1, 5))
            correct_top1.append(prec1.cpu().numpy())
            correct_top5.append(prec5.cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        correct_top5 = sum(correct_top5) / len(correct_top5)
        # correct = correct.float() / len(self.testloader.dataset)
        print("Test set: Average  Accuracy top1:", correct_top1,"top5",correct_top5)
        return correct_top1,correct_top5

    def eval_training(self):
        self.net.eval()
        # test_loss = 0.0  # cost function error
        correct_top1 = []
        correct_top5 = []
        for (inputs, labels) in self.testloader:

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)

                prec1, prec5 = self.accuracy(outputs.data[:, self.start_num:self.end_num],
                                         labels.cuda().data, topk=(1, 5))
            correct_top1.append(prec1.cpu().numpy())
            correct_top5.append(prec5.cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        correct_top5 = sum(correct_top5) / len(correct_top5)
        # correct = correct.float() / len(self.testloader.dataset)
        print("Test set: Average  Accuracy top1:", correct_top1, "top5", correct_top5)
        return correct_top1, correct_top5


    def train(self):
        best_acc,_ = self.eval_training()
        print("init test acc:", best_acc)
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            self.net.train()

            start_time = time.time()
            running_loss = 0.0
            correct = 0.0

            if epoch > self.warm:
                self.train_scheduler.step(epoch)
            for i, (inputs, labels) in enumerate(self.trainloader):
                # get the inputs; data is a list of [inputs, labels]

                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # inputs,  labels = torch.autograd.Variable(inputs),  torch.autograd.Variable(labels)

                outputs = self.net(inputs)
                # preds = outputs.masked_select(targets_one_hot.eq(1))

                tar_ce = labels
                pre_ce = outputs.clone()

                pre_ce = pre_ce[:, self.start_num:self.end_num]

                loss = torch.nn.functional.cross_entropy(pre_ce, tar_ce)
                running_loss += loss.item()
                # prec1, prec5 = self.accuracy(outputs.data[:, self.start_num:self.end_num], labels.cuda().data,
                #                              topk=(1, 5))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            print("epoch", epoch, " time loss", (end_time - start_time))
            print('train set: Average loss: {:.4f},'.format(
                running_loss / len(self.trainloader.dataset),
            ), "best_acc:", best_acc, "lr:", self.train_scheduler.get_lr())
            curr_acc,_ = self.eval_training()
            if curr_acc > best_acc:
                best_acc = curr_acc
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                torch.save(self.net.state_dict(),self.main_net_path)

        print('Finished Training')

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""

        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




