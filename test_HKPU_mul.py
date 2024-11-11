## 홍콩 TEST
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from utility.datasetutil import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

import copy
import sys
import argparse

import timm
join = os.path.join

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csvpath', type=str)
parser.add_argument('--origin_path', type=str)
parser.add_argument('--test_path', type=str)

parser.add_argument('--savepath', type=str)
parser.add_argument('--savenm', type=str)
parser.add_argument('--savenm_total', type=str)

parser.add_argument('--model_path', type=str)

args = parser.parse_args()



#################### debugging용 ###################
# 24.11.11 HKDB-1 test에 맞춰 변경함
args.csvpath = '/content/Datasets/DS_TEST_bad(B)/' 
args.origin_path = '/content/Datasets/images/t2_kj/'  
args.test_path = '/content/drive/MyDrive/perceptual/output/HKdb-1/test_result/'  
args.savepath = '/content/drive/MyDrive/perceptual/output/HKdb-1/foreer/'  
args.savenm = 'test.csv'
args.savenm_total = 'testrttt.csv'
args.model_path = '/content/Output/HKPU_B/checkpoints/'

# # 24.11.08 HKdb-2
# args.csvpath = '/content/Datasets/DS_TEST_bad(A)/' 
# args.origin_path = '/content/Datasets/images/t1_kj/' 
# args.test_path = '/content/drive/MyDrive/perceptual/output/HKdb-2/test_result/' 
# args.savepath = '/content/drive/MyDrive/perceptual/output/HKdb-2/foreer/' 
# args.savenm = 'test.csv'
# args.savenm_total = 'testrttt.csv'
# args.model_path = '/content/Output/HKPU_A/checkpoints/'  

#################### debugging용 ###################

plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
#device = torch.device("cpu")

# authentic imopster csv 경로
authcsvname=args.csvpath + 'registerds.csv'
impcsvname=args.csvpath + 'imposterds.csv'

# authentic and imposter csv에 매칭되는 이미지 경로
path=args.origin_path

# test 이미지 경로 (blended test set)

# test 이미지 경로 (origin test set ==> 학습여부 검증용)
# test 이미지 경로 (origin test set ==> 학습여부 검증용)
# testimgpath = 'Datasets/images/SD_DB1_original_shift_matching/'

# output 경로 및 이름
# real_bad / total_bad ( blended set 결과 )
# real_origin / total_origin ( 원본 set 결과 )
# real_generated / total_generated ( 복원 set 결과 )

savecsvname=args.savenm
savecsvname_for_d=args.savenm_total

#156 ==> HKPU
#318 ==> SDU
numofcls = 156
numofclsfile = 5

#데이터셋 list형태로 호출
auth_ds=authentic_ds(authcsvname)

#trained model load
model_ft = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)
num_ftrs = model_ft.head.fc.in_features
model_ft.head.fc = nn.Linear(num_ftrs, 2)
torch.nn.init.xavier_uniform_(model_ft.head.fc.weight, gain=0.031)

tform=transforms.Compose([
        transforms.ToTensor()
    ])

#test (make csv output)
def test_model(model, epoch, testimgpath, savecsvpath):
    print('TEST START')
    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    impo_ds = imposter_test_ds(impcsvname, path, numofcls, numofclsfile)
    tds = auth_ds + impo_ds  # Datasets/ ...

    # tds = auth_ds
    # tds = impo_ds
    # pytorch dataset

    #0~1
    train_ds = FingerveinDataset_test_zeros(tds,testimgpath,transform=tform,Use_blendset=False)     # 0 ~ 1로 정규화
    #-1~1
    #train_ds = FingerveinDataset_test(tds, transform=tform)  # 0 ~ 1로 정규화
    dataloader = DataLoader(train_ds, batch_size=1, num_workers=4)
    totallen = len(tds)

    # Iterate over data.
    with torch.no_grad():
        for i, datas in enumerate(dataloader):
            labels, inputs, filepath, matching_files = datas
            inputs = torch.reshape(inputs,(inputs.shape[1],3,224,224))
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs=np.array(outputs.tolist())

            loweset_val = outputs[np.argmin(np.abs(outputs[:, 1]))].tolist()

            labels=labels.tolist()

            # 검증용 전체 shift matching set score 저장
            outputs = outputs.tolist()
            for m_idx,data in enumerate(matching_files):
                writecsv(savecsvpath + str(epoch) + savecsvname_for_d,
                         [labels[0], filepath[0][0], matching_files[m_idx][0], outputs[m_idx][0], outputs[m_idx][1]])

            # 실제 EER 계산시 사용할 score 저장
            writecsv(savecsvpath+str(epoch)+savecsvname,
                     [labels[0],filepath[0][0],filepath[1][0],loweset_val[0],loweset_val[1]])


            if i%5==0 and i>0:
                print(str(i) + '/' + str(totallen) + ' Finished!')
            elif i==totallen-1:
                print(str(i) + '/' + str(totallen) + ' Finished!')

if __name__ == '__main__':
    modelpath = args.model_path

    for i in range(3,4):  # 3 EPCOH의 가중치 가져옴
        model_ft.load_state_dict(torch.load(modelpath+str(i)+'EPOCH.pt'))
        model_ft = model_ft.to(device)
        # print(model_ft.eval())

        epoch_list = list(range(350, 750, 50))  # 24.11.11 HKDB-1 test에 맞춰 변경함

        for epoch in epoch_list:
            print(epoch, "epoch start!")
            testimgpath = join(args.test_path, f'epoch_{epoch}_shift_ver2/')
            savecsvpath = join(args.savepath, f'epoch_{epoch}/')

            # 경로가 존재하지 않으면 생성
            if not os.path.exists(savecsvpath):
                os.makedirs(savecsvpath)

            test_model(model_ft, i, testimgpath, savecsvpath)