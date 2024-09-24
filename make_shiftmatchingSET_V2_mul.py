### 3-way shift matching set 만들기 ###

import cv2
import numpy as np
from PIL import  Image
from PIL import  ImageEnhance
from utility.datasetutil import *
import os
import shutil
join = os.path.join

#region augmentaiton 기준 ==> gan training 이후에 training set이 아닌 shift matching 용 데이터를 늘려야 하기 때문에 사용

# 3번째 논문의 목적상 기존과 다르게 상 하 좌 우 센터 대각 포함 9개 이미지를 25%씩 cropping하는 것이 더 적합함

#endregion

## output_path
# for i in range(700, 1220, 20):

base_path = '/content/drive/MyDrive/output/SDdb-1/test_result'  # 24.09.23 SDdb-1

def save_img(img,fol,file):
    v_output = img.astype('uint8')
    im = Image.fromarray(v_output)
    im.save(output_path +fol+'/'+file+'.bmp')
def make_aug_set(only_train=False):
    for i, data in enumerate(ds):
        paths = split(data)

        src = cv2.imread(data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        directorys = paths[0].split('\\')
        mkdir(output_path + directorys[2])

        # 지정맥 이미지는 Hue(색상) ,Saturation(채도), Value(명도) 가운데 Value만 값이 있음
        # 색상과 채도가 없는경우에는 굳이 HSV 로 바꾸지 않고 Gray로 읽어도 같은 값
        h, s, v = cv2.split(img)


        ## 좌 센터 우 (비율은 안맞음)
        save_img(v[:, :160], directorys[2], '1_' + paths[1][2:])

        save_img(v[:, 32:160], directorys[2], '2_' + paths[1][2:])

        save_img(v[:, 32:], directorys[2],'3_' + paths[1][2:])

epoch_list = list(range(250, 550, 50))  # 24.09.23 SDdb-1

for epoch in epoch_list:
    ds = glob(join(base_path, f'epoch_{epoch}'), '*/*', True)  # 해당 경로에 복원된 images(Test images) 넣기
    output_path = join(base_path, f'epoch_{epoch}_shift_ver2/')
    make_aug_set(only_train=False)
