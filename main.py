'''Train CIFAR10 with PyTorch.'''

# Cifar-10 dataset을 closed-set으로 학습을 시키고 SVHN test dataset을 openset으로 Test하는 코드입니다.
# SVHN 데이터셋은 검색해보시면 어떠한 데이터셋인지 나올 겁니다.



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn



import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


import os
import argparse

from models import *
from utils import progress_bar

import numpy as np
import copy

from sklearn import metrics

from cb_loss import CB_loss


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--total_epoch', default=100, type=int, help='total epoch')
parser.add_argument('--interval', default=10, type=int, help="save interval")
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_cls', default=2, type=int, help="num classes")
parser.add_argument('--temperature', default=1.0, type=float, help="temperature scaling")
parser.add_argument('--flood_level', default=0.0, type=float, help="flood level")
parser.add_argument('--ensemble', default=1, type=int, help="ensemble test mode")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([

    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_crop = transforms.Compose([
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
])



# -----------------------------------------------------------------------------------
batch_size=128
num_classes =args.num_cls
T = args.temperature
flood = args.flood_level
total_epoch= args.total_epoch
interval = args.interval
ensemble=args.ensemble


trainset = torchvision.datasets.ImageFolder(
    root='./data/custom2/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset ,batch_size=256, shuffle=True,num_workers=0)


testset = torchvision.datasets.ImageFolder(
    root='./data/custom2/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=0)


cropset = torchvision.datasets.ImageFolder(
    root='./data/custom2/test', transform=transform_crop)
croploader = torch.utils.data.DataLoader(
    cropset, batch_size=256, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------------


# 학습 Model을 정의하는 부분입니다. Resnet18을 사용하겠습니다.

print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()



net = models.resnet18(pretrained=True)
net.fc =nn.Linear(512,num_classes)


net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


# 저장된 모델을 load하는 부분입니다.
# ----------------------------------------------------------------------------------
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
# ----------------------------------------------------------------------------------



# loss function 및 optimizaer, learning rate scheduler를 정의하는 부분입니다.
# -------------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# --------------------------------------------------------------------------------------


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_sample_per_cls(trainloader,num_classes):

    train_sample_fname = trainloader.dataset.samples
    train_sample_fname = np.asarray([np.asarray(i) for i in train_sample_fname])
    sample_per_cls=[]

    for i in range(num_classes):
        sample_per_cls.append((train_sample_fname==str(i)).sum())

    return np.asarray(sample_per_cls)




def train(epoch):
    print('\nEpoch: %d' % epoch)
    print("Current lr : {}".format(get_lr(optimizer)))
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    sample_per_cls=get_sample_per_cls(trainloader, num_classes)


    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs ,targets = inputs.to(device) ,targets.to(device)

        # print(targets)
        optimizer.zero_grad()
        outputs = net(inputs)

        _,cbloss = CB_loss(labels=targets, logits=outputs, T=T, flood=flood, samples_per_cls=sample_per_cls, no_of_classes=num_classes ,loss_type="softmax", beta=0.9, gamma=2.0)
        # loss = criterion(outputs, targets)


        # loss.backward()
        cbloss.backward()
        optimizer.step()

        train_loss += cbloss.item()
        # train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# test 하는 함수입니다.
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    sample_per_cls = get_sample_per_cls(trainloader, num_classes)

    pred_all = []
    target_all = []

    #test loader에 있는 모든 파일의 파일명을 불러옴.
    test_sample_fname = np.asarray(testloader.dataset.samples)



    with torch.no_grad():
        batch_count=0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            probs, cbloss = CB_loss(labels=targets, logits=outputs, T=T, flood=flood, samples_per_cls=sample_per_cls,
                             no_of_classes=num_classes, loss_type="softmax", beta=0.9, gamma=2.0)
            # loss = criterion(outputs, targets)

            test_loss += cbloss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pred_all.extend(predicted.data.cpu().numpy())
            target_all.extend(targets.data.cpu().numpy())



            #---------------------------------------------------------------------------
            # 아래는 예측과 실제 정답이 맞는지 틀린지 확인하는 코드 입니다.
            # print(predicted.eq(targets))

            # 결과 중에 틀린 경우(False)의 index를 출력합니다.
            # print(torch.where(predicted.eq(targets)==False))

            # False index를 numpy형태로 저장 (추후 전체 파일명이 저장된 numpy에서 추려내기 위함)
            false_index = torch.where(predicted.eq(targets)==False)[0].cpu().numpy()

            if len(false_index)==0:
                print("there are no false indexes")
            else:
                false_probs=probs[false_index]
                print(false_probs.max(dim=1)[0])

            # batch마다 전체 파일이 담기지 않기 때문에 batch가 지난 만큼의 file수를 조정하여 index계산
            # (이 부분은 헷갈리시면 말씀해주세요)
            false_index = batch_count+false_index

            print("batch idx : {}".format(batch_idx))
            print("false index len :{}".format(len(false_index)))
            print("false index: {}".format(false_index))

            if len(false_index)==0: #false가 없는 경우
                # print("there are no false indexes")
                pass
            else: #false가 있는 경우 false 파일의 이름 및 길이 출력
                false_file_name = test_sample_fname[false_index]
                # print(false_file_name)
                # print(len(false_file_name))

            batch_count += inputs.size(0) # index 조정을 위한 현재까지 Loading한 file개수 저장
            #---------------------------------------------------------------------------


            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        print("Closed-Set Confusion Matrix")
        print(metrics.confusion_matrix(target_all, pred_all, labels=range(num_classes)))
        print("T :{} , F :{}".format(T,flood))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    if ensemble==True:
        if (epoch+1) % interval==0:
            print("Save model for ensemble test , epoch :{}".format(epoch))
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch' : epoch,
            }
            if not os.path.isdir('./checkpoint/ensemble'):
                os.makedirs('./checkpoint/ensemble')

            torch.save(state, './checkpoint/ensemble/ckpt_interval_{}.pth'.format(int((epoch+1)/interval)))

    return acc




def ensemble_test(total_epoch,interval):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint/ensemble'), 'Error: no checkpoint directory found!'

    #저장된 모델들의 파라메터를 담는 list입니다.
    checkpoint_list=[]

    #for문을 돌면서 checkpoint을 load하고 list에 저장해줍니다.
    for i in range(1,int(total_epoch/interval)+1):
        checkpoint= torch.load('./checkpoint/ensemble/ckpt_interval_{}.pth'.format(i))
        checkpoint_list.append(checkpoint)
        # net.load_state_dict(checkpoint['net'])
        # net_list.append(net)



    # print(checkpoint_list)
    # for i in range(int(total_epoch/interval)):
    #     print(checkpoint_list[i]['acc'])



    softmax = nn.Softmax(dim=1)


    # 저장된 여러개의 모델들의 결과를 저장합니다. (10개 모델이면 10개의 prediction 저장)
    results=[]
    all_targets=[]
    with torch.no_grad():
        for checkpoint in checkpoint_list:
            cur_result=[]
            net=models.resnet18(pretrained=True)
            net.fc=nn.Linear(512,num_classes)
            net=net.to(device)
            net.load_state_dict(checkpoint['net'])
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs=inputs.cuda()
                net.eval()
                outputs = net(inputs)
                probs = softmax(outputs)
                cur_result.extend(probs.cpu().numpy())

            cur_result=np.asarray(cur_result)
            results.append(cur_result)

        for batch_idx, (inputs, targets) in enumerate(testloader):
            all_targets.extend(targets.cpu().numpy())

    all_targets=np.asarray(all_targets)


    results=np.asarray(results)
    # print(results)
    # print(results.shape)
    #
    # print(np.mean(results,axis=0).shape)
    #
    # print(results[:,0].shape)
    # print(results[:,1])

    #o_model은 original_model의 줄임말이며, 최종학습 모델을 의미합니다.
    #e_model은 ensemble model의 줄임말이며, 앙상블 모델을 의미합니다.

    o_model_probs=results[-1] #최종 모델의 결과만 가지고 옵니다.
    e_model_probs=np.mean(results, axis=0) # 모든 모델의 결과를 평균냅니다. (앙상블)

    o_model_conf = np.max(o_model_probs, axis=1) #최종 모델의 confidence 값을 가져옵니다.
    e_model_conf = np.max(e_model_probs, axis=1) #앙상블 모델의 confidence값을 가져옵니다.

    o_model_preds= np.argmax(o_model_probs, axis=1) #최종 모델의 prediction 값을 가져옵니다.
    e_model_preds= np.argmax(e_model_probs, axis=1) #앙상블 모델의 prediction 값을 가져옵니다.

    o_model_false_index = np.where(np.equal(all_targets,o_model_preds)==False) #최종 모델의 틀린 예측의 인덱스를 가져옵니다.
    e_model_false_index = np.where(np.equal(all_targets,e_model_preds)==False) #앙상블 모델의 틀린 예측의 인덱스를 가져옵니다.

    o_model_true_index = np.where(np.equal(all_targets, o_model_preds) == True) #최종 모델의 맞춘 예측의 인덱스를 가져옵니다.
    e_model_true_index = np.where(np.equal(all_targets, e_model_preds) == True) #앙상블 모델의 맞춘 예측의 인덱스를 가져옵니다.


    #--------------------------------------------------------------------------------------------------------------------------
    #최종 모델과 , 앙상블 모델에서 틀린 sample에 대한 confidence값을 비교합니다.
    #이때, 최종모델에서 틀린 sample들만 비교하는 코드입니다.
    #Confidence가 좀 내려가긴 하는 듯 합니다.


    print(o_model_false_index)
    print(o_model_conf[o_model_false_index])
    print("Original false mean:{} , var :{}".format(np.mean(o_model_conf[o_model_false_index]),np.var(o_model_conf[o_model_false_index])))

    print(e_model_false_index)
    print(e_model_conf[o_model_false_index])
    print("Ensemble false mean:{} , var :{} (o_model false index)".format(np.mean(e_model_conf[o_model_false_index]),
                                              np.var(e_model_conf[o_model_false_index])))

    print("")
    print("")

    #---------------------------------------------------------------------------------------------------------------------------




    #------------------------------------------------------------------------------------------------------
    #최종 모델과 앙상블 모델간의 맞춘 sample들의 confidence 분포와, 틀린 sample들의 confidence값을 비교합니다.
    #최종 모델보다는 앙상블 모델이 틀린 sample에 대한 confidence 값이 전반적으로 낮았습니다.
    #그러나 틀린 sample의 개수가 조금 증가합니다.  (저의 경우는 한개)
    #학습 epoch늘리면 이 부분을 해결될듯 합니다.

    print("Original true mean:{} , var :{}".format(np.mean(o_model_conf[o_model_true_index]),
                                                    np.var(o_model_conf[o_model_true_index])))

    print("Original false mean:{} , var :{}".format(np.mean(o_model_conf[o_model_false_index]),
                                                    np.var(o_model_conf[o_model_false_index])))

    print("Ensemble true mean:{} , var :{}".format(np.mean(e_model_conf[e_model_true_index]),
                                                   np.var(e_model_conf[e_model_true_index])))

    print("Emsemble false mean:{} , var :{}".format(np.mean(e_model_conf[e_model_false_index]),
                                                    np.var(e_model_conf[e_model_false_index])))

    print("")
    print(e_model_conf[e_model_false_index])
    #------------------------------------------------------------------------------------------------------




def crop_test(trial):
    print("10 Crop ensemble test")
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


    softmax = nn.Softmax(dim=1)


    crop_probs=[]
    all_probs=[]
    o_probs=[]

    all_targets= []



    with torch.no_grad():
        for i in range(trial):
            all_probs=[]
            for batch_idx , (inputs, targets) in enumerate(croploader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                probs = softmax(outputs)
                all_probs.extend(probs.cpu().numpy())

            all_probs=np.asarray(all_probs)
            crop_probs.append(all_probs)

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            probs = softmax(outputs)
            o_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())



    crop_probs=np.asarray(crop_probs)
    o_probs=np.asarray(o_probs)
    all_targets=np.asarray(all_targets)

    print(crop_probs.shape)
    print(o_probs.shape)
    print(all_targets.shape)



    crop_probs= np.mean(crop_probs,axis=0)

    crop_confidence = np.max(crop_probs, axis=1)
    crop_pred = np.argmax(crop_probs, axis=1)

    o_confidence = np.max(o_probs, axis=1)
    o_pred = np.argmax(o_probs, axis=1)

    false_index_o = np.where(np.equal(all_targets,o_pred)==False)
    false_index_crop = np.where(np.equal(all_targets,crop_pred)==False)

    # print(crop_confidence[false_index_o])
    print(false_index_o)
    print(o_confidence[false_index_o])
    print("o_conf mean: {}, var: {}".format(np.mean(o_confidence[false_index_o]), np.var(o_confidence[false_index_o])))

    print(false_index_crop)
    print(crop_confidence[false_index_crop])
    print("crop_conf mean: {}, var: {}".format(np.mean(crop_confidence[false_index_crop]), np.var(crop_confidence[false_index_crop])))

    # print(o_confidence[false_index_crop])




if __name__=='__main__':
    # 실제 코드 실행하는 부분입니다.
    past_acc=0
    for epoch in range(start_epoch, start_epoch+total_epoch):
        train(epoch) #train 함수 호출
        cur_acc=test(epoch)  #test 함수 호출
        acc_diff= abs(cur_acc-past_acc)
        past_acc=copy.deepcopy(cur_acc)

        print("acc diff: {:5f}".format(acc_diff))
        scheduler.step()

    ensemble_test(total_epoch,interval)
    # crop_test(20)
