######################### 학습  ################################
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision

import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision import datasets, models, transforms
from utility.datasetutil import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import copy
import timm

authcsvname = 'Datasets/DS_t2_aug/registerds.csv'
impcsvname = 'Datasets/DS_t2_aug/imposterds.csv'
path = 'Datasets/images/t2_aug/'
checkpoint_path = 'Output/HKPU_B/checkpoints/'
plt.ion()  # interactive mode
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

# HKPU 기준 156
# SDU 기준 318
numofcls = 156

# class 기준으로 900개
numofclsfile = 30

# 데이터셋 list형태로 호출
auth_ds = authentic_ds(authcsvname)

# log (tensorboard)
writer = SummaryWriter('Output/HKPU_B/summaries')

# 데이터셋을 tensor type으로 변경 하고 normalize
tform = transforms.Compose([
     transforms.ToTensor()
])


# train 함수 (pytorch tutorial 참조)
def train_model(model, criterion, optimizer, scheduler, num_epochs=10, startingpoint=0):
    since = time.time()
    # 최초 모델 가중치 copy
    best_model_wts = copy.deepcopy(model.state_dict())  # for last epoch
    best_model_wts_in_iteration = copy.deepcopy(model.state_dict())  # for every iteration
    best_acc = 0.0
    iternum = 0

    for epoch in range(startingpoint, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        # model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Dataset setting(imposter 데이터 갯수가 authentic 보다 많기 때문에 두 클래스의 숫자를 동일하게 맞춰주기 위함)
        impo_ds = imposter_ds(impcsvname, path, numofcls, numofclsfile)
        tds = auth_ds + impo_ds
        # pytorch dataset
        train_ds = FingerveinDataset_zeros(tds, transform=tform)
        dataloader = DataLoader(train_ds, batch_size=4, shuffle=True)

        loss_before = 0.0

        # Iterate over data.
        for i, datas in enumerate(dataloader):
            labels, inputs, _ = datas

            #test (이미지 확인용)
            '''
            d1 = inputs[0].permute(1, 2, 0).numpy()
            d2 = inputs[1].permute(1, 2, 0).numpy()
            d3 = inputs[2].permute(1, 2, 0).numpy()
            d4 = inputs[3].permute(1, 2, 0).numpy()

            fig = plt.figure(figsize=(30, 30))
            plt.subplot(2, 8, 1)
            plt.imshow(d1)

            plt.subplot(2, 8, 2)
            plt.imshow(d2)

            plt.subplot(2, 8, 3)
            plt.imshow(d3)

            plt.subplot(2, 8, 4)
            plt.imshow(d4)

            plt.show()
            '''
            #test
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model(inputs)

            loss = criterion(outputs,
                             labels)  ## pytorch의 crossentropy loss는 logsoftmax() 함수가 포함된 형태이므로 바로 backward 처리하고 accuracy를 보고 싶으면 따로 softmax function 으로 처리 하여 계산
            loss.backward()
            optimizer.step()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            # backward + optimize only if in training phase

            # 현재 loss & accuracy
            now_loss = loss.item()
            now_acc = torch.sum(preds == labels.data) / 4.0
            ###추가###
            # 현재 iteration LOSS 기준 제일 낮은 iteration LOSS를 따로 저장하기 위해 loss 비교 및 copy

            if loss_before == 0.0:  # 시작 iteration LOSS를 받아둠
                loss_before = now_loss
            else:
                if now_loss < loss_before:  # 시작시점 이외에 이전 LOSS가 현재 LOSS보다 작은 경우 weight deep copy 이후 이전 loss update
                    best_model_wts_in_iteration = copy.deepcopy(model.state_dict())
                    loss_before = now_loss

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            print(str(4 * (i + 1)) + '/' + str(len(tds)) + ' finish!')

            iternum += 4

        optimizer_params = optimizer.state_dict().popitem()
        learning_rate = optimizer_params[1][0]['lr']
        writer.add_scalar('learning rate',
                          learning_rate,
                          epoch)

        scheduler.step()
        torch.save(best_model_wts_in_iteration, checkpoint_path + str(epoch) + '_EPOCH_lowest_loss.pt')
        epoch_loss = running_loss / len(tds)
        epoch_acc = running_corrects.double() / len(tds)

        writer.add_scalar('training accuracy',
                          epoch_acc,
                          epoch)
        writer.add_scalar('training loss',
                          epoch_loss,
                          epoch)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'training', epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()
        torch.save(model.state_dict(), checkpoint_path + str(epoch) + 'EPOCH.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# trained model load
model_ft = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)
# Weight Freezing
'''
ct = 0
for child in model_ft.children():
    ct += 1
    if ct < 2:
        for param in child.parameters():
            param.requires_grad = False
'''
num_ftrs = model_ft.head.fc.in_features
model_ft.head.fc = nn.Linear(num_ftrs, 2)
torch.nn.init.xavier_uniform_(model_ft.head.fc.weight, gain=0.031)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()


# Observe that all parameters are being optimized
# starting lr = 0.0001
### 빌드시에 lr 꼭확인하기
optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.00001,betas=[0.9,0.999])
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# 가장 최근 가중치 가져오도록 수정필요 (현재 하드코딩)
#modelpath = 'Output/checkpoints/3EPOCH.pt'
#model_ft.load_state_dict(torch.load(modelpath))

# train
best_weight_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=4, startingpoint=0)
# save best mode
torch.save(best_weight_model.state_dict(), checkpoint_path + 'best_weight_model_in_10epoch.pt')