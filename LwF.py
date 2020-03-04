
import torch

from torchvision import transforms
from torch.utils import data
import time
import os
import numpy as np
from ResNet import resnet18
from cifar100_split import Cifar100Split

from models import DecoderNet

import torch.nn.functional as F

import copy
import utils
from cifar10_svnh import Cifar10_SVNH_Split
class Trainer(object):
    def __init__(self, start_num=0, end_num=10, net=None,save_path="",data_name="cifar100",epoch=50):
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
        self.main_net_path = save_path+"/Lwf_"+str(start_num)+".ptn"
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
            trainset = Cifar100Split(start_num=start_num, end_num=end_num, train=True, transform=transform_train)
            testset = Cifar100Split(start_num=start_num, end_num=end_num, train=False, transform=transform_test)
        elif data_name is "cifar10":
            trainset = Cifar10_SVNH_Split(isCifar10=True, start_num=start_num, end_num=end_num, train=True,
                                          transform=transform_train)
            testset = Cifar10_SVNH_Split(isCifar10=True, start_num=start_num, end_num=end_num, train=False,
                                         transform=transform_test)
        else:
            trainset = Cifar10_SVNH_Split(isCifar10=False, start_num=start_num, end_num=end_num, train=True,
                                          transform=transform_train)
            testset = Cifar10_SVNH_Split(isCifar10=False, start_num=start_num, end_num=end_num, train=False,
                                         transform=transform_test)
        self.trainloader = data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.testloader = data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=False)
        milestones=[10, 20, 30, 40]
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
        if start_num>0:
            self.distillation=True
            # self.old_model = copy.deepcopy(net)
        else:
            self.distillation=False
            # self.old_model = None

    def load_model(self,model_path=""):

        prev_best = torch.load(model_path)
        self.net.load_state_dict(prev_best)
        self.old_model = copy.deepcopy(self.net)
        self.old_model.eval()
        print("old")

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

                prec1= self.accuracy(outputs.data[:, start_num:end_num],
                                             labels.cuda().data, topk=(1,))
            correct_top1.append(prec1[0].cpu().numpy())
            # correct_top5.append(prec5.cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        # correct_top5 = sum(correct_top5) / len(correct_top5)
        # correct = correct.float() / len(self.testloader.dataset)
        print("Test set: Average  Accuracy top1:")
        return correct_top1

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

                prec1 = self.accuracy(outputs.data[:, self.start_num:self.end_num],
                                         labels.cuda().data, topk=(1,))
            correct_top1.append(prec1[0].cpu().numpy())
            # correct_top5.append(prec5.cpu().numpy())
        correct_top1 = sum(correct_top1) / len(correct_top1)
        # correct_top5 = sum(correct_top5) / len(correct_top5)
        # correct = correct.float() / len(self.testloader.dataset)
        print("Test set: Average  Accuracy top1:", correct_top1)
        return correct_top1

    def _train_decoder(self,i,inputs):
        self.decoder_list[i].zero_grad()
        self.decoder_list[i].train()
        with torch.no_grad():
            features,_ = self.net.forward_feature(inputs)
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
                    if len(tem)>0:
                        loss_pixel = self._train_decoder(i,tem)
                        running_loss_g += loss_pixel
            end_time = time.time()
            print("epoch", epoch, " time loss", (end_time - start_time))
            # print("train set: Average loss", running_loss_g)
            pix_loss = self.eval_decoder()
            for i in range(self.class_num):
                if self.best_loss_pix[i] > pix_loss[i]:
                    self.best_loss_pix[i] = pix_loss[i]
                    print("best pix loss ",i,":",self.best_loss_pix[i])
                    if not os.path.exists(self.save_path):
                            os.mkdir(self.save_path)
                    save_path=self.save_path+"/decoder_"+str(self.start_num+i)+".ptn"
                    torch.save(self.decoder_list[i].state_dict(),save_path)

    def eval_decoder(self):
        loss_pix = {}
        for i in range(self.class_num):
            loss_pix[i] =0

        for _, (images, labels) in enumerate(self.testloader):
            images = images.cuda()
            self.net.eval()
            with torch.no_grad():
                for i in range(self.class_num):
                    tem = images[labels.eq(i)]
                    if len(tem) > 0:
                        features,_ = self.net.forward_feature(tem)
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
            correct = 0.0

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
                    # batch_in =inputs[:1000]
                    # print(batch_in.shape)
                    # outputs_old = self.old_model(batch_in, path, -1)
                    # targets_one_hot = torch.FloatTensor(inputs.shape[0], self.start_num).cuda()
                    # targets_one_hot.zero_()
                    # targets_one_hot.scatter_(1, labels[:, None], 1)

                    with torch.no_grad():
                        outputs_old = self.old_model(inputs)

                    # t_one_hot = targets_one_hot.clone()
                    t_one_hot= outputs_old[:,0:self.start_num]

                    # cx = self.args.rigidness_coff * (self.args.sess - self.args.jump)
                    # loss_dist =  torch.mean(F.kl_div(F.log_softmax(outputs[:,0:self.start_num] / 5.0, dim=1), F.softmax(t_one_hot / 5.0, dim=1),
                    #              reduce=None).clamp(min=0.0))
                    loss_dist = F.binary_cross_entropy(F.softmax(outputs[:, 0:self.start_num] / 2.0, dim=1),
                                                    F.softmax(t_one_hot / 2.0, dim=1))


                loss += 10*loss_dist
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
            curr_acc= self.eval_training()
            if curr_acc > best_acc:
                best_acc = curr_acc
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
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

def model_test(model_num=5, save_path="", per_num=10):

    alpha = np.exp(-0.02 * model_num * per_num)
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    testset = Cifar100Split(start_num=0, end_num=model_num * 10, train=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    decoder_list = []
    net = resnet18(num_classes=100).cuda()
    model_path = torch.load(save_path+"/LwF_"+str((model_num-1)*10)+".ptn")
    net.load_state_dict(model_path)
    net.eval()
    feature_extract_list=[]
    for j in range(model_num):
        feature_extract = resnet18(num_classes=100).cuda()
        model_path = torch.load(save_path+"/LwF_" + str(j*10) + ".ptn")
        feature_extract.load_state_dict(model_path)
        feature_extract.eval()
        feature_extract_list.append(feature_extract)
    for j in range(model_num * 10):
        de = DecoderNet().cuda()
        model_path = torch.load(save_path+"/decoder_" + str(j) + ".ptn")
        de.load_state_dict(model_path)
        de.eval()
        decoder_list.append(de)

    correct_top1 = 0
    correct_rec = 0
    correct_combination = 0
    over_num = 0
    over_num2 = 0
    for n, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            output = net(images)
            preds = output[:, 0:model_num * 10]
            _, pre_label_top1 = preds.max(dim=1)
            correct_top1 += pre_label_top1.eq(labels).sum()
            k = model_num+1
            pre_score, pre_label = preds.topk(k=k, dim=1)
            for i in range(images.size(0)):
                score = preds[i, labels[i]]
                over_confident = preds[i] > score
                over_num += over_confident.sum()

                # print(correct)
                pix_loss = torch.zeros([k]).cuda()
                for j in range(len(pre_label[i])):
                    path = pre_label[i][j]
                    task = path//10
                    feature,_ = feature_extract_list[task].forward_feature(images[i:i+1])
                    rec_img = decoder_list[path](feature)
                    a = images[i].view(-1)
                    b = rec_img.view(-1)
                    loss = torch.mean(torch.abs(a - b))
                    # pix_loss[j]= 0.1*(mean_pix_loss[path]-loss)
                    pix_loss[j] = loss

                # print(pix_loss)
                if pre_label[i][pix_loss.argmin()] == labels[i]:
                    correct_rec += 1
                    # print(correct_rec)
                a = (pre_score[i] - min(pre_score[i])) / (max(pre_score[i]) - min(pre_score[i]))
                a = torch.nn.functional.softmax(a, dim=0)
                pix_loss = (pix_loss - min(pix_loss)) / (max(pix_loss) - min(pix_loss))
                b = torch.nn.functional.softmax(-pix_loss, dim=0)
                c = alpha * a + (1 - alpha) * b
                if pre_label[i][c.argmax()] == labels[i]:
                    correct_combination += 1

                if (pre_label[i].eq(labels[i])).sum() > 0:
                    index = torch.where(pre_label[i] == labels[i])
                    score = c[index]
                    a = c > score
                    over_num2 += a.sum()
                else:
                    over_num2 += k

    num_data = len(testloader.dataset)
    correct_combination = correct_combination / num_data
    correct_rec = correct_rec / num_data
    correct_top1 = correct_top1.float() / num_data
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec, " alpha:", alpha)

    log = "Test model:" + str(model_num) + " Accuracy predict+decoder:" + str(
        correct_combination) + " only predict top1:" + str(
        correct_top1) + "noly rec min1:" + str(correct_rec) + ""

    utils.save_acc_csv(save_file="/over_confident100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=over_num.cpu().numpy(), model_name="LwF_100-" + str(per_num))
    utils.save_acc_csv(save_file="/over_confident100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=over_num2.cpu().numpy(), model_name="LwF_Decoder100-" + str(per_num))

    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=correct_top1.cpu().numpy(), model_name="LwF_100-" + str(per_num))
    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=correct_combination, model_name="LwF_Decoder100-" + str(per_num))
    with open(save_path + "/log.txt", "a+", encoding="utf-8") as fp:
        fp.write(log + "\n")
        fp.close()





if __name__ == '__main__':

    per_num = 2
    num_class = 10
    data_name = "svnh"
    save_path = "model/" + data_name + "/LwF" + str(per_num)
    for i in range(0, int(num_class/per_num)):
        torch.cuda.empty_cache()
        net = resnet18(num_classes=num_class).cuda()
        trainer = Trainer(net=net, start_num=i * per_num, end_num=(i + 1) * per_num, save_path=save_path,data_name=data_name)
        if i>0:
            previous_model = save_path+"/LwF_" + str((i-1) * per_num) + ".ptn"
            trainer.load_model(model_path=previous_model)
        trainer.train()
        curr_model = torch.load(save_path+"/LwF_" + str(i * per_num) + ".ptn")
        net.load_state_dict(curr_model)
        trainer.trainDecoder()



