
import torch

from torchvision import transforms
from torch.utils import data
import time
import os
import numpy as np
from ResNet import resnet18
from cifar100_split import Cifar100Split
from cifar100_rehearsal import Cifar100Rehearsal

from models import DecoderNet

import torch.nn.functional as F

import copy
from cifar10_svnh import Cifar10_SVNH_Split, Cifar10_SVNH_Rehearsal


class Trainer(object):
    def __init__(self, start_num=0, end_num=10, rehearsal_size=2000, net=None, save_path="", data_name="cifar100",
                 epoch=50):
        self.lr = 0.001
        self.epoch = epoch
        self.warm = 1
        self.batch_size = 256
        self.start_num = start_num
        self.end_num = end_num
        self.class_num = end_num - start_num
        self.use_cuda = True
        self.task_num = 1
        self.save_path = save_path
        self.main_net_path = save_path + "/icarl_" + str(start_num) + ".ptn"
        self.rehearsal_size = rehearsal_size
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

        if data_name is "cifar100":
            self.trainset = Cifar100Split(start_num=start_num, end_num=end_num, train=True, transform=transform_train)
            self.testset = Cifar100Split(start_num=start_num, end_num=end_num, train=False, transform=transform_test)
        elif data_name is "cifar10":
            self.trainset = Cifar10_SVNH_Split(isCifar10=True, start_num=start_num, end_num=end_num, train=True,
                                               transform=transform_train)
            self.testset = Cifar10_SVNH_Split(isCifar10=True, start_num=start_num, end_num=end_num, train=False,
                                              transform=transform_test)
        else:
            self.trainset = Cifar10_SVNH_Split(isCifar10=False, start_num=start_num, end_num=end_num, train=True,
                                               transform=transform_train)
            self.testset = Cifar10_SVNH_Split(isCifar10=False, start_num=start_num, end_num=end_num, train=False,
                                              transform=transform_test)
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.rehearsal_data = None
        self.rehearsal_loader = None
        if start_num > 0:
            self.rehearsal_data = Cifar100Rehearsal(end_num=start_num, rehearsal_size=2000, transform=transform_train)
            if data_name is "cifar100":
                self.rehearsal_data = Cifar100Rehearsal(end_num=start_num, rehearsal_size=2000,
                                                        transform=transform_train)
            elif data_name is "cifar10":
                self.rehearsal_data = Cifar10_SVNH_Rehearsal(isCifar10=True,end_num=start_num, rehearsal_size=2000,
                                                             transform=transform_train)
            else:
                self.rehearsal_data = Cifar10_SVNH_Rehearsal(isCifar10=False,end_num=start_num, rehearsal_size=2000,
                                                             transform=transform_train)
            self.rehearsal_loader = data.DataLoader(self.rehearsal_data, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=0)

        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=False)
        milestones = [10, 20, 30, 40]
        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.5)
        self.optmizer_De_list = []
        self.scheduler_De_list = []
        self.decoder_list = []
        self.pixelwise_loss = torch.nn.L1Loss()
        self.best_loss_pix = {}

        for i in range(self.class_num):
            self.best_loss_pix[i] = 1000000
            de = DecoderNet()
            if self.use_cuda:
                de = de.cuda()
            self.decoder_list.append(de)
            opt = torch.optim.Adam(de.parameters(), lr=self.lr, betas=(0.9, 0.999))
            self.optmizer_De_list.append(opt)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)
            self.scheduler_De_list.append(scheduler)
        self.old_model = None
        if start_num > 0:
            self.distillation = True
            # self.old_model = copy.deepcopy(net)
        else:
            self.distillation = False
            # self.old_model = None

    def load_model(self, model_path=""):

        prev_best = torch.load(model_path)
        self.net.load_state_dict(prev_best)
        self.old_model = copy.deepcopy(self.net)
        self.old_model.eval()
        print("old")

    def test_acc(self, data_loader, start_num, end_num):
        self.net.eval()
        # test_loss = 0.0  # cost function error
        correct_top1 = []
        for (inputs, labels) in data_loader:

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)

                prec1 = self.accuracy(outputs.data[:, start_num:end_num],
                                      labels.cuda().data, topk=(1,))
            correct_top1.append(prec1[0].cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        # correct = correct.float() / len(self.testloader.dataset)
        print("Test set: Average  Accuracy top1:", correct_top1)
        return correct_top1

    def eval_training(self):
        self.net.eval()
        # test_loss = 0.0  # cost function error
        correct_top1 = []
        for (inputs, labels) in self.testloader:

            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.net(inputs)

                prec1= self.accuracy(outputs.data[:, self.start_num:self.end_num],
                                             labels.cuda().data, topk=(1,))
            correct_top1.append(prec1[0].cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        # correct = correct.float() / len(self.testloader.dataset)
        print("Test set: Average  Accuracy top1:", correct_top1)
        return correct_top1

    def _train_decoder(self, i, inputs):
        self.decoder_list[i].zero_grad()
        self.decoder_list[i].train()
        with torch.no_grad():
            features, _ = self.net.forward_feature(inputs)
        rec_img = self.decoder_list[i](features)
        loss_pixel = self.pixelwise_loss(rec_img, inputs)
        loss_pixel.backward()
        self.optmizer_De_list[i].step()
        return loss_pixel.item()

    def trainDecoder(self):
        self.eval_training()
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            start_time = time.time()
            running_loss_g = 0.0
            if epoch > self.warm:
                for i in range(self.class_num):
                    self.scheduler_De_list[i].step(epoch)
            for _, (inputs, labels) in enumerate(self.trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.cuda()
                for i in range(self.class_num):
                    tem = inputs[labels.eq(i)]
                    if len(tem) > 0:
                        loss_pixel = self._train_decoder(i, tem)
                        running_loss_g += loss_pixel
            end_time = time.time()
            print("epoch", epoch, " time loss", (end_time - start_time))
            # print("train set: Average loss", running_loss_g)
            pix_loss = self.eval_decoder()
            for i in range(self.class_num):
                if self.best_loss_pix[i] > pix_loss[i]:
                    self.best_loss_pix[i] = pix_loss[i]
                    print("best pix loss ", i, ":", self.best_loss_pix[i])
                    if not os.path.exists(self.save_path):
                        os.mkdir(self.save_path)
                    save_path = self.save_path + "/decoder_" + str(self.start_num + i) + ".ptn"
                    torch.save(self.decoder_list[i].state_dict(), save_path)

    def eval_decoder(self):
        loss_pix = {}
        for i in range(self.class_num):
            loss_pix[i] = 0

        for _, (images, labels) in enumerate(self.testloader):
            images = images.cuda()
            self.net.eval()
            with torch.no_grad():
                for i in range(self.class_num):
                    tem = images[labels.eq(i)]
                    if len(tem) > 0:
                        features, _ = self.net.forward_feature(tem)
                        self.decoder_list[i].eval()
                        rec_img = self.decoder_list[i](features)
                        loss_pixel = self.pixelwise_loss(rec_img, tem)
                        loss_pix[i] += loss_pixel
        return loss_pix

    def train(self):
        best_acc = self.eval_training()
        print("init test acc:", best_acc)
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            self.net.train()

            start_time = time.time()
            running_loss = 0.0

            if epoch > self.warm:
                self.train_scheduler.step(epoch)
            for i, (inputs, labels) in enumerate(self.trainloader):
                # get the inputs; data is a list of [inputs, labels]
                # labels = labels +self.start_num
                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # inputs,  labels = torch.autograd.Variable(inputs),  torch.autograd.Variable(labels)
                outputs = self.net(inputs)
                # preds = outputs.masked_select(targets_one_hot.eq(1))
                tar_ce = labels
                pre_ce = outputs.clone()
                pre_ce = pre_ce[:, self.start_num:self.end_num]
                loss = torch.nn.functional.cross_entropy(pre_ce, tar_ce)
                loss_dist = 0
                ## distillation loss
                if self.distillation:
                    with torch.no_grad():
                        outputs_old = self.old_model(inputs)
                    t_one_hot = outputs_old[:, 0:self.start_num]
                    loss_dist = F.binary_cross_entropy(F.softmax(outputs[:, 0:self.start_num] / 2.0, dim=1),
                                                       F.softmax(t_one_hot / 2.0, dim=1))
                loss += loss_dist
                running_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.rehearsal_loader is not None:
                for i, (inputs, labels) in enumerate(self.rehearsal_loader):
                    if self.use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    # inputs,  labels = torch.autograd.Variable(inputs),  torch.autograd.Variable(labels)
                    outputs = self.net(inputs)
                    # preds = outputs.masked_select(targets_one_hot.eq(1))
                    tar_ce = labels
                    pre_ce = outputs[:, 0:self.start_num]
                    loss = torch.nn.functional.cross_entropy(pre_ce, tar_ce)
                    running_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            end_time = time.time()
            print("epoch", epoch, " time loss", (end_time - start_time))
            print('train set: Average loss: {:.4f},'.format(
                running_loss / len(self.trainloader.dataset),
            ), "best_acc:", best_acc, "lr:", self.train_scheduler.get_lr())
            curr_acc = self.eval_training()
            if curr_acc > best_acc:
                best_acc = curr_acc
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                torch.save(self.net.state_dict(), self.main_net_path)

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




if __name__ == '__main__':

    per_num = 2
    num_class = 10
    data_name = "svnh"
    save_path = "model/" + data_name + "/icarl" + str(per_num)
    for i in range(0, int(num_class / per_num)):
        torch.cuda.empty_cache()
        net = resnet18(num_classes=num_class).cuda()
        trainer = Trainer(net=net, start_num=i * per_num, end_num=(i + 1) * per_num, save_path=save_path,
                          data_name=data_name)
        if i > 0:
            previous_model = save_path + "/icarl_" + str((i - 1) * per_num) + ".ptn"
            trainer.load_model(model_path=previous_model)
        trainer.train()
        curr_model = torch.load(save_path + "/icarl_" + str(i * per_num) + ".ptn")
        net.load_state_dict(curr_model)
        trainer.trainDecoder()
