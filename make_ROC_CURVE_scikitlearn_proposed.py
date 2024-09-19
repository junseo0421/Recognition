### 원본 / 이미지 없어진거  / proposed  EER CHART ###

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score

from utility.datasetutil import *

from scipy import interpolate

# 하나씩 불러서 근사화 이후 평균으로 계산
score_path1='Output/HKPU_A/foreer/3real_HKPU_B_t2_kj_original_shift_matching.csv'
score_path2='Output/HKPU_B/foreer/3real_HKPU_A_original_shift_matching.csv'

s2_path1='Output/HKPU_A/foreer/3real_HKPU_B_t2_kj_CROP_W25P_shift_matching.csv'
s2_path2='Output/HKPU_B/foreer/3real_t1_kj_CROP_W25P_shift_matching.csv'

s3_path1='Output/HKPU_A/foreer/3real_HKPU_B_Proposed_1060_shift_ver2.csv'
s3_path2='Output/HKPU_B/foreer/3real_HKPU_A_Proposed_300_shift_ver2.csv'


def make_fpr_tpr(score_path1):
    S1 = np.array(csv2list(score_path1))

    labels = S1[:, 0].astype(np.single)
    score1 = S1[:, 3:5].astype(np.single)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    test_labels=[]
    for i in range(2):
        if i==0:
            input_label = np.where(labels == 0, 1, 0)
            test_labels.append(input_label)
        else:
            input_label = np.where(labels == 1, 1, 0)
            test_labels.append(input_label)
        fpr[i], tpr[i], thr = roc_curve(labels, score1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr,tpr,roc_auc,test_labels,score1

fpr ,tpr ,_ ,_ ,_ = make_fpr_tpr(score_path1)
f1 = interpolate.interp1d(fpr[1], tpr[1])

fpr2 ,tpr2 ,_ ,_ ,_ = make_fpr_tpr(score_path2)
f2 = interpolate.interp1d(fpr2[1], tpr2[1])

fpr3 ,tpr3 ,_ ,_ ,_ = make_fpr_tpr(s2_path1)
f3 = interpolate.interp1d(fpr3[1], tpr3[1])

fpr4 ,tpr4 ,_ ,_ ,_ = make_fpr_tpr(s2_path2)
f4 = interpolate.interp1d(fpr4[1], tpr4[1])

fpr5 ,tpr5 ,_ ,_ ,_ = make_fpr_tpr(s3_path1)
f5 = interpolate.interp1d(fpr5[1], tpr5[1])

fpr6 ,tpr6 ,_ ,_ ,_ = make_fpr_tpr(s3_path2)
f6 = interpolate.interp1d(fpr6[1], tpr6[1])


xnew = np.arange(0, 1, 0.001)

ynew1 = f1(xnew)
ynew2 = f2(xnew)

ynew3 = f3(xnew)
ynew4 = f4(xnew)

ynew5 = f5(xnew)
ynew6 = f6(xnew)


plt.figure()
lw = 1
plt.rc('font', size=15)
plt.rc('legend', fontsize=13)

plt.yticks(np.arange(0,105,5))
plt.xticks(np.arange(0,105,5))

plt.plot(np.arange(100,0,-1), color='black',
         lw=lw, label='EER line')
plt.plot(xnew*100, ((ynew1*100)+(ynew2*100))/2.0, color='b',
         lw=lw, label='Original')
plt.plot(xnew*100, ((ynew3*100)+(ynew4*100))/2.0, color='r',
         lw=lw, label='Without 25% of right or left')
plt.plot(xnew*100, ((ynew5*100)+(ynew6*100))/2.0, color='g',
         lw=lw, label='Restorated')

plt.xlim([0, 30])
plt.ylim([70, 100])
plt.xlabel('FAR (%)')
plt.ylabel('GAR (%)')
plt.legend(loc="lower center")

plt.savefig('EER_HKPU.png')

plt.show()


print('aa')



'''
def make_fpr_tpr(score_path1,score_path2):
    S1 = np.array(csv2list(score_path1))
    S2 = np.array(csv2list(score_path2))

    AllS = np.concatenate((S1, S2), axis=0)

    labels = AllS[:, 0].astype(np.float)
    score1 = AllS[:, 3:5].astype(np.float)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    test_labels=[]
    for i in range(2):
        if i==0:
            input_label = np.where(labels == 0, 1, 0)
            test_labels.append(input_label)
        else:
            input_label = np.where(labels == 1, 1, 0)
            test_labels.append(input_label)
        fpr[i], tpr[i], _ = roc_curve(labels, score1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr,tpr,roc_auc,test_labels,score1

fpr,tpr,roc_auc,test_labels,score1=make_fpr_tpr(score_path1,score_path2)
test_labels=np.array(test_labels)
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), score1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

fpr2,tpr2,roc_auc2,test_labels2,score2=make_fpr_tpr(score_targetpath1,score_targetpath2)
test_labels2=np.array(test_labels2)
# Compute micro-average ROC curve and ROC area
fpr2["micro"], tpr2["micro"], _ = roc_curve(test_labels2.ravel(), score2.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])


plt.figure()
lw = 2
plt.rc('font', size=15)
plt.rc('legend', fontsize=10)

plt.yticks(np.arange(0,105,5))
plt.plot(np.arange(100,0,-1), color='black',
         lw=lw, linewidth=1)
plt.plot(fpr[1]*100, tpr[1]*100, color='b',
         lw=lw,linewidth=1)
plt.plot(fpr2[1]*100, tpr2[1]*100, color='r',
         lw=lw ,linewidth=1)

plt.xlim([0, 5])
plt.ylim([95, 100])
plt.xlabel('FAR (%)')
plt.ylabel('GAR (%)')
#plt.legend(loc="lower right")

plt.savefig('EER_SDU-DB_calculation_origin_closed.png')

plt.show()

print('aaaa')
'''