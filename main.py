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

from sklearn import metrics

from cb_loss import CB_loss


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_cls', default=2, type=int, help="num classes")
parser.add_argument('--temperature', default=1.0, type=float, help="temperature scaling")
parser.add_argument('--flood_level', default=0.0, type=float, help="flood level")
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



# -----------------------------------------------------------------------------------
batch_size=128
num_classes =args.num_cls
T = args.temperature
flood = args.flood_level
#
# sample_per_cls=np.asarray([272,10])
# weights = 1 / torch.Tensor(sample_per_cls)
# weights = weights.double()
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_classes)
# trainloader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, sampler = sampler)



trainset = torchvision.datasets.ImageFolder(
    root='./data/custom/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset ,batch_size=128, shuffle=True,num_workers=0)


testset = torchvision.datasets.ImageFolder(
    root='./data/custom/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=24, shuffle=False, num_workers=0)

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

            # print("batch idx : {}".format(batch_idx))
            # print("false index len :{}".format(len(false_index)))
            # print("false index: {}".format(false_index))

            if len(false_index)==0: #false가 없는 경우
                # print("there are no false indexes")
                pass
            else: #false가 있는 경우 false 파일의 이름 및 길이 출력장
                false_file_name = test_sample_fname[false_index]
                # print(false_file_name)
                # print(len(false_file_name))

            batch_count += inputs.size(0) # index 조정을 위한 현재까지 Loading한 file개수 저
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


if __name__=='__main__':
    #실제 코드 실행하는 부분입니다.
    for epoch in range(start_epoch, start_epoch+300):
        train(epoch) #train 함수 호출
        test(epoch)  #test 함수 호출

        # 앞서 보았던 evaludate_openset함수를 실행하고 output인 auroc값을 출력
        # 이때 입력으로는 network, closed-testloader, open-testloader를 줌.
        scheduler.step()
