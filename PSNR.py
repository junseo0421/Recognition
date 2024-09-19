## PSNR 계산

import numpy as np
import math
import cv2

from utility.datasetutil import *

paths1 = glob('Datasets/images/SD_DB1_original/', '*/0_*')  # center image만 가져옴
paths2 = glob('Datasets/images/SD_DB2_original/', '*/0_*')  # center image만 가져옴

Apath = paths1 + paths2

paths3 = glob('Datasets/images/Proposed_V2/SDU_A_Proposed_440', '*/0_*')  # center image만 가져옴
paths4 = glob('Datasets/images/Proposed_V2/SDU_B_Proposed_380', '*/0_*')  # center image만 가져옴

Bpath = paths3 + paths4

psnr_ls = []
for i in range(len(Apath)):
    ori_img = cv2.imread(Apath[i])
    ori_img = cv2.resize(ori_img, dsize=(180, 70), interpolation=cv2.INTER_LINEAR)
    con_img = cv2.imread(Bpath[i])
    con_img = cv2.resize(con_img, dsize=(180, 70), interpolation=cv2.INTER_LINEAR)

    ori_max = np.max(ori_img)
    con_max = np.max(con_img)

    max_pixel = max([ori_max, con_max])

    # MSE 계산
    mse = np.mean((ori_img - con_img) ** 2)

    if mse == 0:
        psnr_ls.append(100)
    else:
        # PSNR 계산
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        psnr_ls.append(psnr)
print(sum(psnr_ls) / len(psnr_ls))