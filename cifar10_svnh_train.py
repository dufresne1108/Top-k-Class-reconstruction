
import torch

from torchvision import transforms
from torch.utils import data

import numpy as np
from ResNet import resnet18

from models import DecoderNet

from cifar10_svnh import Cifar10_SVNH_Split
from FixedRepresentation import Trainer as fixed_trainer
from LwF import Trainer as lwf_trainer
from icarl import Trainer as icarl_trainer
from FineTuning import Trainer as finetune_trainer


def train_fixedRep(data_name="cifar100", per_num=2, num_class=10, epoch=50):
    save_path = "model/" + data_name + "/FixedRepresentation" + str(per_num)
    net = resnet18(num_classes=num_class).cuda()
    for i in range(0, int(num_class / per_num)):
        torch.cuda.empty_cache()
        if i > 0:
            prev_best = torch.load(save_path + "/FixedRep.ptn")
            net.load_state_dict(prev_best)
        trainer = fixed_trainer(net=net, start_num=i * per_num, end_num=(i + 1) * per_num, save_path=save_path,
                                data_name=data_name, epoch=epoch)
        trainer.train()
        prev_best = torch.load(save_path + "/FixedRep.ptn")
        net.load_state_dict(prev_best)
        trainer.trainDecoder()


def train_lwf(data_name="cifar100", per_num=2, num_class=10, epoch=50):
    save_path = "model/" + data_name + "/LwF" + str(per_num)
    for i in range(0, int(num_class / per_num)):
        torch.cuda.empty_cache()
        net = resnet18(num_classes=num_class).cuda()
        trainer = lwf_trainer(net=net, start_num=i * per_num, end_num=(i + 1) * per_num, save_path=save_path,
                              data_name=data_name, epoch=epoch)
        if i > 0:
            previous_model = save_path + "/LwF_" + str((i - 1) * per_num) + ".ptn"
            trainer.load_model(model_path=previous_model)
        trainer.train()
        curr_model = torch.load(save_path + "/LwF_" + str(i * per_num) + ".ptn")
        net.load_state_dict(curr_model)
        trainer.trainDecoder()


def train_fineTune(data_name="cifar100", per_num=2, num_class=10, epoch=50):
    save_path = "model/" + data_name + "/FineTuning" + str(per_num)
    for i in range(0, int(num_class / per_num)):
        torch.cuda.empty_cache()
        net = resnet18(num_classes=num_class).cuda()
        trainer = finetune_trainer(net=net, start_num=i * per_num, end_num=(i + 1) * per_num, save_path=save_path,
                                   data_name=data_name, epoch=epoch)
        if i > 0:
            previous_model = save_path + "/FineTuning_" + str((i - 1) * per_num) + ".ptn"
            trainer.load_model(model_path=previous_model)
        trainer.train()
        curr_model = torch.load(save_path + "/FineTuning_" + str(i * per_num) + ".ptn")
        net.load_state_dict(curr_model)
        trainer.trainDecoder()


def train_icarl(data_name="cifar100", per_num=2, num_class=10, epoch=50):
    save_path = "model/" + data_name + "/icarl" + str(per_num)
    for i in range(0, int(num_class / per_num)):
        torch.cuda.empty_cache()
        net = resnet18(num_classes=num_class).cuda()
        trainer = icarl_trainer(net=net, start_num=i * per_num, end_num=(i + 1) * per_num, save_path=save_path,
                                data_name=data_name, epoch=epoch)
        if i > 0:
            previous_model = save_path + "/icarl_" + str((i - 1) * per_num) + ".ptn"
            trainer.load_model(model_path=previous_model)
        trainer.train()
        curr_model = torch.load(save_path + "/icarl_" + str(i * per_num) + ".ptn")
        net.load_state_dict(curr_model)
        trainer.trainDecoder()


def model_test_FixedRep(model_num=5, save_path="", per_num=10, alpha=0.02, num_class=100, testloader=None):
    # 测试模型
    alpha = np.exp(-alpha * model_num * per_num)
    torch.cuda.empty_cache()

    decoder_list = []
    net = resnet18(num_classes=num_class).cuda()
    prev_best = torch.load(save_path + "/FixedRep.ptn")
    net.load_state_dict(prev_best)
    net.eval()
    for j in range(model_num * per_num):
        de = DecoderNet().cuda()
        prev_best = torch.load(save_path + "/decoder_" + str(j) + ".ptn")
        de.load_state_dict(prev_best)
        de.eval()
        decoder_list.append(de)
    correct_top1 = 0
    correct_rec = 0
    correct_combination = 0
    for n, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()
        # print(n)
        with torch.no_grad():
            feature, output7 = net.forward_feature(images)
            preds = output7[:, 0:model_num * per_num]
            _, pre_label_top1 = preds.max(dim=1)
            correct_top1 += pre_label_top1.eq(labels).sum()
            k = model_num + 1
            pre_score, pre_label = preds.topk(k=k, dim=1)
            for i in range(images.size(0)):

                pix_loss = torch.zeros([k]).cuda()
                for j in range(len(pre_label[i])):
                    path = pre_label[i][j]
                    rec_img = decoder_list[path](feature[i:i + 1])
                    a = images[i].view(-1)
                    b = rec_img.view(-1)
                    loss = torch.mean(torch.abs(a - b))
                    pix_loss[j] = loss

                if pre_label[i][pix_loss.argmin()] == labels[i]:
                    correct_rec += 1
                a = (pre_score[i] - min(pre_score[i])) / (max(pre_score[i]) - min(pre_score[i]))
                a = torch.nn.functional.softmax(a, dim=0)
                pix_loss = (pix_loss - min(pix_loss)) / (max(pix_loss) - min(pix_loss))
                b = torch.nn.functional.softmax(-pix_loss, dim=0)
                c = alpha * a + (1 - alpha) * b
                if pre_label[i][c.argmax()] == labels[i]:
                    correct_combination += 1

    num_data = len(testloader.dataset)
    correct_combination = correct_combination / num_data
    correct_rec = correct_rec / num_data
    correct_top1 = correct_top1.float() / num_data
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha)


def model_test_(model_num=5, save_path="", per_num=10, alpha=0.02, model_name="", num_class=100, testloader=None):
    # 测试模型
    alpha = np.exp(-alpha * model_num * per_num)
    torch.cuda.empty_cache()

    decoder_list = []
    net = resnet18(num_classes=num_class).cuda()
    model_path = torch.load(save_path + "/" + model_name + "_" + str((model_num - 1) * per_num) + ".ptn")
    net.load_state_dict(model_path)
    net.eval()
    feature_extract_list = []
    for j in range(model_num):
        feature_extract = resnet18(num_classes=num_class).cuda()
        model_path = torch.load(save_path + "/" + model_name + "_" + str(j * per_num) + ".ptn")
        feature_extract.load_state_dict(model_path)
        feature_extract.eval()
        feature_extract_list.append(feature_extract)
    for j in range(model_num * per_num):
        de = DecoderNet().cuda()
        model_path = torch.load(save_path + "/decoder_" + str(j) + ".ptn")
        de.load_state_dict(model_path)
        de.eval()
        decoder_list.append(de)

    correct_top1 = 0
    correct_rec = 0
    correct_combination = 0

    for n, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            output = net(images)
            preds = output[:, 0:model_num * per_num]
            _, pre_label_top1 = preds.max(dim=1)
            correct_top1 += pre_label_top1.eq(labels).sum()
            k = model_num + 1
            pre_score, pre_label = preds.topk(k=k, dim=1)
            for i in range(images.size(0)):

                # print(correct)
                pix_loss = torch.zeros([k]).cuda()
                for j in range(len(pre_label[i])):
                    path = pre_label[i][j]
                    task = path // per_num
                    feature, _ = feature_extract_list[task].forward_feature(images[i:i + 1])
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

    num_data = len(testloader.dataset)
    correct_combination = correct_combination / num_data
    correct_rec = correct_rec / num_data
    correct_top1 = correct_top1.float() / num_data
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha)


def model_test():
    data_name = ["cifar10", "svnh"]
    per_num = 2
    num_class = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    for item in data_name:

        if item is "svnh":
            testset = Cifar10_SVNH_Split(isCifar10=False, start_num=0, end_num=10, train=False,
                                         transform=transform_test)
            print("test data: ", item)
        else:
            testset = Cifar10_SVNH_Split(isCifar10=True, start_num=0, end_num=10, train=False,
                                         transform=transform_test)
            print("test data: ", item)
        testloader = data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)
        print("test model: FixedRepresentation")
        save_path = "model/" + item + "/FixedRepresentation" + str(per_num)
        model_test_FixedRep(model_num=5, save_path=save_path, per_num=per_num, alpha=0.1, num_class=num_class,
                            testloader=testloader)
        print("test model: FineTuning")
        save_path = "model/" + item + "/FineTuning" + str(per_num)
        model_test_(model_num=5, save_path=save_path, per_num=per_num, alpha=0.1, model_name="fineTuning",
                    num_class=num_class, testloader=testloader)
        print("test model: icarl")
        save_path = "model/" + item + "/icarl" + str(per_num)
        model_test_(model_num=5, save_path=save_path, per_num=per_num, alpha=0.01, model_name="icarl",
                    num_class=num_class, testloader=testloader)
        print("test model: LwF")
        save_path = "model/" + item + "/LwF" + str(per_num)
        model_test_(model_num=5, save_path=save_path, per_num=per_num, alpha=0.1, model_name="Lwf",
                    num_class=num_class,
                    testloader=testloader)


def train_model():
    data_name = ["cifar10", "svnh"]
    per_num = 2
    num_class = 10
    for item in data_name:
        train_lwf(data_name=item, per_num=per_num, num_class=num_class, epoch=50)
        train_icarl(data_name=item, per_num=per_num, num_class=num_class, epoch=50)
        train_fineTune(data_name=item, per_num=per_num, num_class=num_class, epoch=50)
        train_fixedRep(data_name=item, per_num=per_num, num_class=num_class, epoch=50)


if __name__ == '__main__':
    train_model()
    # model_test()
