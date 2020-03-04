import utils
import torch
from torchvision import transforms
from torch.utils import data
import numpy as np
from ResNet import resnet18
from cifar100_split import Cifar100Split
from models import DecoderNet
def model_test_FixedRep(model_num=5, save_path="", per_num=10,alpha=0.02,num_class=100):

    alpha = np.exp(-alpha * model_num * per_num)
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    testset = Cifar100Split(start_num=0, end_num=model_num * per_num, train=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
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
    over_num = 0
    over_num2 = 0
    for n, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            feature, output7 = net.forward_feature(images)
            preds = output7[:, 0:model_num * per_num]
            _, pre_label_top1 = preds.max(dim=1)
            correct_top1 += pre_label_top1.eq(labels).sum()
            k = model_num + 1
            pre_score, pre_label = preds.topk(k=k, dim=1)
            for i in range(images.size(0)):
                score = preds[i, labels[i]]
                over_confident = preds[i] > score
                over_num += over_confident.sum()


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
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha)
    utils.save_acc_csv(save_file="/over_confident100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=over_num.cpu().numpy(), model_name="FixedRep_100-" + str(per_num))
    utils.save_acc_csv(save_file="/over_confident100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=over_num2.cpu().numpy(), model_name="FixedRep_Decoder100-" + str(per_num))

    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=correct_top1.cpu().numpy(), model_name="FixedRep_100-" + str(per_num))
    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=correct_combination, model_name="FixedRep_Decoder100-" + str(per_num))


def model_test(model_num=5, save_path="", per_num=10,model_name="",alpha=0.02):

    alpha = np.exp(-alpha* model_num * per_num)
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    testset = Cifar100Split(start_num=0, end_num=model_num * per_num, train=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    decoder_list = []
    net = resnet18(num_classes=100).cuda()
    model_path = torch.load(save_path + "/"+model_name+"_" + str((model_num - 1) * per_num) + ".ptn")
    net.load_state_dict(model_path)
    net.eval()
    feature_extract_list = []
    for j in range(model_num):
        feature_extract = resnet18(num_classes=100).cuda()
        model_path = torch.load(save_path + "/"+model_name+"_" + str(j * per_num) + ".ptn")
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
    over_num = 0
    over_num2 = 0
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
                score = preds[i, labels[i]]
                over_confident = preds[i] > score
                over_num += over_confident.sum()

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
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha)

    utils.save_acc_csv(save_file="/over_confident100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=over_num.cpu().numpy(), model_name=model_name+"_100-" + str(per_num))
    utils.save_acc_csv(save_file="/over_confident100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=over_num2.cpu().numpy(), model_name=model_name+"_Decoder100-" + str(per_num))

    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=correct_top1.cpu().numpy(), model_name=model_name+"_100-" + str(per_num))
    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + ".csv", class_num=model_num * per_num,
                       acc=correct_combination, model_name=model_name+"_Decoder100-" + str(per_num))

def model_test_FixedRep_decay(model_num=5, save_path="", per_num=10,alpha1=0.02,more_result=False):

    alpha = np.exp(-alpha1 * model_num * per_num)
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    testset = Cifar100Split(start_num=0, end_num=model_num * per_num, train=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    decoder_list = []
    net = resnet18(num_classes=100).cuda()
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
    over_num = 0
    over_num2 = 0
    for n, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            feature, output7 = net.forward_feature(images)
            preds = output7[:, 0:model_num * per_num]
            _, pre_label_top1 = preds.max(dim=1)
            correct_top1 += pre_label_top1.eq(labels).sum()
            k = model_num + 1
            pre_score, pre_label = preds.topk(k=k, dim=1)
            for i in range(images.size(0)):
                score = preds[i, labels[i]]
                over_confident = preds[i] > score
                over_num += over_confident.sum()


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
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha1)
    if more_result is True:
        utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + "decay.csv", class_num=model_num * per_num,
                           acc=correct_top1.cpu().numpy(), model_name="only FixedRep")
        utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + "decay.csv", class_num=model_num * per_num,
                            acc=correct_rec, model_name="only Decoder")

    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + "decay.csv", class_num=model_num * per_num,
                       acc=correct_combination, model_name="FixedRep_Decoder-" + str(alpha1))

def model_lwf_decay_test(model_num=5, save_path="", per_num=10,model_name="",alpha1=0.02):

    alpha = np.exp(-alpha1* model_num * per_num)
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    testset = Cifar100Split(start_num=0, end_num=model_num * per_num, train=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    decoder_list = []
    net = resnet18(num_classes=100).cuda()
    model_path = torch.load(save_path + "/"+model_name+"_" + str((model_num - 1) * per_num) + ".ptn")
    net.load_state_dict(model_path)
    net.eval()
    feature_extract_list = []
    for j in range(model_num):
        feature_extract = resnet18(num_classes=100).cuda()
        model_path = torch.load(save_path + "/"+model_name+"_" + str(j * per_num) + ".ptn")
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
    over_num = 0
    over_num2 = 0
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
                score = preds[i, labels[i]]
                over_confident = preds[i] > score
                over_num += over_confident.sum()

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
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha)

    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + "decay.csv", class_num=model_num * per_num,
                       acc=correct_combination, model_name="LwF_Decoder-" + str(alpha1))
def task_decay_test():
    per_num =10
    save_path = "model/FixedRepresentation"
    alpha_list=[0.1,0.01,0.001,0.0001]
    for al in alpha_list:
        if al is 0.1:
            re=True
        else:
            re =False
        for i in range(int(100 / per_num)):
            model_test_FixedRep_decay(model_num=i + 1, save_path=save_path, per_num=per_num, alpha1=al,more_result=re)




def model_k_FixedRep(tk=1,model_num=5, save_path="", per_num=10,alpha=0.02):

    alpha = np.exp(-alpha * model_num * per_num)
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    testset = Cifar100Split(start_num=0, end_num=model_num * per_num, train=False, transform=transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    decoder_list = []
    net = resnet18(num_classes=100).cuda()
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
    over_num = 0
    over_num2 = 0
    for n, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            feature, output7 = net.forward_feature(images)
            preds = output7[:, 0:model_num * per_num]
            _, pre_label_top1 = preds.max(dim=1)
            correct_top1 += pre_label_top1.eq(labels).sum()
            k = tk
            pre_score, pre_label = preds.topk(k=k, dim=1)
            for i in range(images.size(0)):
                score = preds[i, labels[i]]
                over_confident = preds[i] > score
                over_num += over_confident.sum()


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
    print("Test model:", str(model_num), "  Average  Accuracy:", correct_top1, "rec acc", correct_rec,
          " Accuracy predict+decoder:", correct_combination, " alpha:", alpha)
    utils.save_acc_csv(save_file="/decoder_acc100-" + str(per_num) + "k.csv", class_num=tk,
                       acc=correct_combination, model_name="FixedRep_Decoder100-" + str(per_num))

def task_k_test():
    per_num =10
    save_path = "model/FixedRepresentation"
    k_list=[2,5,10,15,20,30,40,70,90]
    for al in k_list:

        model_k_FixedRep(tk=al,model_num=10, save_path=save_path, per_num=per_num)


if __name__ == '__main__':
    # task_decay_test()
    task_k_test()
    # model_test()

