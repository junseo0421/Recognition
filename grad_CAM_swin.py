## transformer GradCAM

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
import os
import copy

import os.path as osp

import timm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

### 56 28 14 7
def reshape_transform1(tensor, height=56, width=56):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform2(tensor, height=28, width=28):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform3(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform4(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



plt.ion()  # interactive mode
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

numofcls=156
numofclsfile=5


#trained model load
# model_ft = timm.create_model('convnext_small_384_in22ft1k', pretrained=True)
model_ft = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)
num_ftrs = model_ft.head.fc.in_features
model_ft.head.fc = nn.Linear(num_ftrs, 2)
torch.nn.init.xavier_uniform_(model_ft.head.fc.weight, gain=0.031)

tform=transforms.Compose([
        transforms.ToTensor()
    ])

func_list =[reshape_transform1,reshape_transform2,reshape_transform3,reshape_transform4]
def test_model(model,epoch):
    print('TEST START')
    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    # pytorch dataset
    # enrolled images
    I1_ls = glob('Datasets/GRADCAM_INPUT1', '*.bmp')
    # target images
    I2_ls = glob('Datasets/GRADCAM_INPUT2', '*.bmp')

    ## composite image 저장 경로
    compo_path = 'GRAD_CAM_OUTPUT/COMPOSITE_IMAGE/'
    ## gradCAM 저장 경로
    output_dir = 'GRAD_CAM_OUTPUT/GRADCAM_IMAGE'

    # output 에 표시할 class
    classes = ['authentic', 'imposter']

    #
    train_ds = FingerveinDataset_test_zeros_FOR_GRADCAM(I1_ls, I2_ls, compo_path, 0, tform)  # 0 ~ 1로 정규화

    dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
    totallen=len(I1_ls)
    # Iterate over data.
    for i,(input, label) in enumerate(dataloader):

        input = input.to(device)
        # GradCam으로 뽑아낼 target layer 4개
        target_layers = [model.layers[0].blocks[-1].norm2, model.layers[1].blocks[-1].norm2, model.layers[2].blocks[-1].norm2, model.layers[3].blocks[-1].norm2]
        # target_layers = [model.layers[0].blocks[-1].norm2,model.layers[1].blocks[-1].norm2,model.layers[2].blocks[-1].norm2,model.layers[3].blocks[-1].norm2]
        # target_layers = [model.layers[0].blocks[-1].drop_path2, model.layers[1].blocks[-1].drop_path2, model.layers[2].blocks[-1].drop_path2, model.layers[3].blocks[-1].drop_path2]

        for k in range(3):
            gcam = GradCAM(model=model, target_layers=[target_layers[k]], reshape_transform=func_list[k])

            targets = [ClassifierOutputTarget(label.item())]

            grayscale_cam = gcam(input_tensor=input,
                                targets=targets,
                                eigen_smooth=True,
                                aug_smooth=False)
            grayscale_cam = torch.from_numpy(grayscale_cam)
            outputs = torch.nn.functional.softmax(grayscale_cam, dim=2)
            outputs -= outputs.min(dim=1, keepdim=True)[0]
            outputs /= outputs.max(dim=1, keepdim=True)[0]
            origin_img = (input[0].permute(1, 2, 0).cuda()).detach().cpu().numpy()
            # cam_image = show_cam_on_image(origin_img, outputs.numpy())
            save_gradcam(str(i)+'_'+str(k)+'.jpg',outputs[0],origin_img,paper_cmap=True)

modelpath='Output/HKPU_A/checkpoints/'
for i in range(3,4):
    #if i in check_array :
        model_ft.load_state_dict(torch.load(modelpath+str(i)+'EPOCH.pt'))
        model_ft = model_ft.to(device)
        print(model_ft.eval())
        test_model(model_ft,i)

