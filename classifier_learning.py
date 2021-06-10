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

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics

from cb_loss import CB_loss

#from torchsampler import ImbalancedDatasetSampler

import cv2


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_cls', default=2, type=int, help="num classes")
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
    testset, batch_size=100, shuffle=False, num_workers=0)

# --------------------------------------------------------------------------------

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

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


lamda= 1

net = models.resnet18(pretrained=True)
net.fc =nn.Linear(512,num_classes)


net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


# 저장된 모델을 load하는 부분입니다.
# ----------------------------------------------------------------------------------

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


def get_sample_per_cls(trainset,num_classes):
    sample_per_cls=np.zeros(num_classes)
    for i in range(num_classes):
        classset=torch.utils.data.Subset(trainset, i)
        classloader=torch.utils.data.DataLoader(classset,batch_size=1,shuffle=False,num_workers=0)
        for idx , _ in enumerate(classloader):
            sample_per_cls[i]+=1


    return sample_per_cls




def train(epoch):
    print('\nEpoch: %d' % epoch)
    print("Current lr : {}".format(get_lr(optimizer)))
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # sample_per_cls=get_sample_per_cls(trainset,num_classes)
    # print(sample_per_cls)
    #
    # exit()


    # Classifier빼고 모두 freeze
    cnt=0
    for param in net.parameters():
        param.requires_grad = False
        cnt += 1
        if cnt==61:
            break

    # classifier만 학습 확인
    # for param in net.parameters():
    #     print(param)



    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs ,targets = inputs.to(device) ,targets.to(device)

        # print(targets)
        optimizer.zero_grad()
        outputs = net(inputs)

        #ok, ng sample 수 입력하시면 됩니다.
        sample_per_cls=np.asarray([272,15])
        cbloss = CB_loss(targets,outputs,sample_per_cls,2,"softmax",0.9,2.0)
        loss = criterion(outputs, targets)


        # loss.backward()
        cbloss.backward()
        optimizer.step()

        train_loss += cbloss.item()
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

    pred_all = []
    target_all = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pred_all.extend(predicted.data.cpu().numpy())
            target_all.extend(targets.data.cpu().numpy())


            # #---------------------------------------------------------------------------
            # # 아래는 예측과 실제 정답이 맞는지 틀린지 확인하는 코드 입니다.
            # print(predicted.eq(targets))
            #
            # # 결과 중에 틀린 경우(False)의 index를 출력합니다.
            # print(torch.where(predicted.eq(targets)==False))
            # #---------------------------------------------------------------------------

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        print("Closed-Set Confusion Matrix")
        print(metrics.confusion_matrix(target_all, pred_all, labels=range(num_classes)))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/stage2'):
            os.mkdir('checkpoint/stage2')
        torch.save(state, './checkpoint/stage2/ckpt.pth')
        best_acc = acc


if __name__=='__main__':
    #실제 코드 실행하는 부분입니다.
    for epoch in range(start_epoch, start_epoch+300):
        train(epoch) #train 함수 호출
        test(epoch)  #test 함수 호출

        # 앞서 보았던 evaludate_openset함수를 실행하고 output인 auroc값을 출력
        # 이때 입력으로는 network, closed-testloader, open-testloader를 줌.
        scheduler.step()
