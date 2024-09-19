## 원본 이미지로 9way shift matching set 만들기 ##
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from utility.datasetutil import *
import os
import shutil

#region augmentaiton 기준 ==> gan training 이후에 training set이 아닌 shift matching 용 데이터를 늘려야 하기 때문에 사용



###    아래의 예시 기준은 홍콩 DB, 산동 DB 모두 동일

# training set은 복원 set 그대로 사용  Test시에는 shift matching 포함한 set으로 진행
# augmentation 네이밍 기준 ==> xx는 고정 앞의 넘버링만 바뀜
# 0_x_x_xx ==> center image
# 1_x_x_xx ==> 이미지를 오른쪽으로 3pixel 만큼 밀고 왼쪽 3pixel mirror padding image
# 2_x_x_xx ==> 이미지를 왼쪽으로 3pixel 만큼 밀고 오른쪽 3pixel mirror padding image
# 3_x_x_xx ==> 이미지를 아래쪽으로 7pixel 만큼 밀고 위쪽 7pixel mirror padding image
# 4_x_x_xx ==> 1번 3번 mix (shift matching용)
# 5_x_x_xx ==> 2번 3번 mix (shift matching용)
# 6_x_x_xx ==> 이미지를 위쪽으로 7pixel 만큼 밀고 아래쪽 7pixel mirror padding image
# 7_x_x_xx ==> 1번 6번 mix (shift matching용)
# 8_x_x_xx ==> 2번 6번 mix (shift matching용)
#endregion

join = os.path.join
## output_path
output_path = 'Datasets/images/SD_DB2_original_shift_matching_TRAIN/'
ds = glob(join('Datasets/images', 'SD_DB2_original'), '*/*', True)

def save_img(img,fol,file):
    v_output = img.astype('uint8')
    im = Image.fromarray(v_output)
    im.save(output_path + fol + '/' + file +'.bmp')

def hconcat(data,iter_num):
    rdata=data
    for i in range(iter_num-1):
        rdata=cv2.hconcat([rdata,data])
    return rdata

def vconcat(data,iter_num):
    rdata = data
    for i in range(iter_num-1):
        rdata=cv2.vconcat([rdata,data])
    return rdata


def make_aug_set(only_train=True):
    for i,data in enumerate(ds):
            paths = split(data)
        #if paths[1][0]=="0":
            ## 이미지 read 이후 hsv 변환 및 분리
            src = cv2.imread(data, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)


            # 지정맥 이미지는 Hue(색상) ,Saturation(채도), Value(명도) 가운데 Value만 값이 있음
            # 색상과 채도가 없는경우에는 굳이 HSV 로 바꾸지 않고 Gray로 읽어도 같은 값
            h, s, v = cv2.split(img)

            up = v[0:1, 0:v.shape[1]]
            up_cut = v[7:v.shape[0],0:v.shape[1]]

            down = v[v.shape[0]-1:v.shape[0], 0:v.shape[1]]
            down_cut = v[0:v.shape[0]-7,0:v.shape[1]]

            left = v[0:v.shape[0], 0:1]
            left_cut = v[0:v.shape[0], 20:v.shape[1]]

            right= v[0:v.shape[0],v.shape[1]-1:v.shape[1]]
            right_cut = v[0:v.shape[0], 0:v.shape[1]-20]

            left=hconcat(left,20)
            right = hconcat(right,20)
            up = vconcat(up,7)
            down = vconcat(down,7)


            directorys = paths[0].split('\\')
            mkdir(output_path + directorys[2])

            left_aug=cv2.hconcat([left,right_cut])
            right_aug=cv2.hconcat([left_cut,right])
            up_aug=cv2.vconcat([up,down_cut])
            down_aug=cv2.vconcat([up_cut,down])

            shutil.copyfile(data, output_path + directorys[2] + '/' + paths[1] + paths[2])

            save_img(left_aug, directorys[2], '1_'+paths[1][2:])

            save_img(right_aug, directorys[2], '2_'+paths[1][2:])

            save_img(up_aug, directorys[2], '3_'+paths[1][2:])

            save_img(down_aug, directorys[2], '6_'+paths[1][2:])

            if only_train==False:
                left_up_cut = left_aug[7:left_aug.shape[0], 0:left_aug.shape[1]]
                left_down_cut = left_aug[0:left_aug.shape[0]-7, 0:left_aug.shape[1]]

                right_up_cut = right_aug[7:right_aug.shape[0], 0:right_aug.shape[1]]
                right_down_cut = right_aug[0:right_aug.shape[0]-7, 0:right_aug.shape[1]]

                left_up_aug = cv2.vconcat([left_up_cut,down])
                left_down_aug = cv2.vconcat([up,left_down_cut ])

                right_up_aug = cv2.vconcat([right_up_cut, down ])
                right_down_aug = cv2.vconcat([up,right_down_cut])

                save_img(left_down_aug, directorys[2], '4_' + paths[1][2:])

                save_img(right_down_aug, directorys[2], '5_' + paths[1][2:])

                save_img(left_up_aug, directorys[2], '7_'+paths[1][2:])

                save_img(right_up_aug, directorys[2], '8_'+paths[1][2:])

make_aug_set(only_train=False)