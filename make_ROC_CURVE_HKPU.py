### 비교실험 HKPU 전체 EER CHART ###

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
import pylab
# 하나씩 불러서 근사화 이후 평균으로 계산
score_path1='Output/HKPU_A/foreer/3real_HKPU_B_ODE_shift_ver2.csv'
score_path2='Output/HKPU_B/foreer/3real_HKPU_A_ODE_shift_ver2.csv'

s2_path1='Output/HKPU_A/foreer/3real_HKPU_B_SN_DCR_shift_ver2.csv'
s2_path2='Output/HKPU_B/foreer/3real_HKPU_A_SN_DCR_shift_ver2.csv'

s3_path1='Output/HKPU_A/foreer/3real_Utrans_HKPU_B_shift_ver2.csv'
s3_path2='Output/HKPU_B/foreer/3real_Utrans_HKPU_A_shift_ver2.csv'

s4_path1='Output/HKPU_A/foreer/3real_QUERY_OTR_HKPU_B_shift_ver2.csv'
s4_path2='Output/HKPU_B/foreer/3real_QUERY_OTR_HKPU_A_shift_ver2.csv'

s5_path1='Output/HKPU_A/foreer/3real_HKPU_B_CUT_shift_ver2.csv'
s5_path2='Output/HKPU_B/foreer/3real_HKPU_A_CUT_shift_ver2.csv'

s6_path1='Output/HKPU_A/foreer/3real_HARMO_HKPU_B_shift_ver2.csv'
s6_path2='Output/HKPU_B/foreer/3real_HARMO_HKPU_A_shift_ver2.csv'

s7_path1='Output/HKPU_A/foreer/3real_P2PHD_HKPU_B_shift_ver2.csv'
s7_path2='Output/HKPU_B/foreer/3real_P2PHD_HKPU_A_shift_ver2.csv'

s8_path1='Output/HKPU_A/foreer/3real_HKPU_B_cyc_shift_ver2.csv'
s8_path2='Output/HKPU_B/foreer/3real_HKPU_A_cyc_shift_ver2.csv'

s9_path1='Output/HKPU_A/foreer/3real_HKPU_B_p2p_shift_ver2.csv'
s9_path2='Output/HKPU_B/foreer/3real_HKPU_A_p2p_shift_ver2.csv'

s10_path1='Output/HKPU_A/foreer/3real_HKPU_B_Proposed_1060_shift_ver2.csv'
s10_path2='Output/HKPU_B/foreer/3real_HKPU_A_Proposed_300_shift_ver2.csv'


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

fpr7 ,tpr7 ,_ ,_ ,_ = make_fpr_tpr(s4_path1)
f7 = interpolate.interp1d(fpr7[1], tpr7[1])

fpr8 ,tpr8 ,_ ,_ ,_ = make_fpr_tpr(s4_path2)
f8 = interpolate.interp1d(fpr8[1], tpr8[1])

fpr9 ,tpr9 ,_ ,_ ,_ = make_fpr_tpr(s5_path1)
f9 = interpolate.interp1d(fpr9[1], tpr9[1])

fpr10 ,tpr10 ,_ ,_ ,_ = make_fpr_tpr(s5_path2)
f10 = interpolate.interp1d(fpr10[1], tpr10[1])

fpr11 ,tpr11 ,_ ,_ ,_ = make_fpr_tpr(s6_path1)
f11 = interpolate.interp1d(fpr11[1], tpr11[1])

fpr12 ,tpr12 ,_ ,_ ,_ = make_fpr_tpr(s6_path2)
f12 = interpolate.interp1d(fpr12[1], tpr12[1])

fpr13 ,tpr13 ,_ ,_ ,_ = make_fpr_tpr(s7_path1)
f13 = interpolate.interp1d(fpr13[1], tpr13[1])

fpr14 ,tpr14 ,_ ,_ ,_ = make_fpr_tpr(s7_path2)
f14 = interpolate.interp1d(fpr14[1], tpr14[1])

fpr15 ,tpr15 ,_ ,_ ,_ = make_fpr_tpr(s8_path1)
f15 = interpolate.interp1d(fpr15[1], tpr15[1])

fpr16 ,tpr16 ,_ ,_ ,_ = make_fpr_tpr(s8_path2)
f16 = interpolate.interp1d(fpr16[1], tpr16[1])

fpr17 ,tpr17 ,_ ,_ ,_ = make_fpr_tpr(s9_path1)
f17 = interpolate.interp1d(fpr17[1], tpr17[1])

fpr18 ,tpr18 ,_ ,_ ,_ = make_fpr_tpr(s9_path2)
f18 = interpolate.interp1d(fpr18[1], tpr18[1])

fpr19 ,tpr19 ,_ ,_ ,_ = make_fpr_tpr(s10_path1)
f19 = interpolate.interp1d(fpr19[1], tpr19[1])

fpr20 ,tpr20 ,_ ,_ ,_ = make_fpr_tpr(s10_path2)
f20 = interpolate.interp1d(fpr20[1], tpr20[1])


xnew = np.arange(0, 1, 0.001)

ynew1 = f1(xnew)
ynew2 = f2(xnew)

ynew3 = f3(xnew)
ynew4 = f4(xnew)

ynew5 = f5(xnew)
ynew6 = f6(xnew)

ynew7 = f7(xnew)
ynew8 = f8(xnew)

ynew9 = f9(xnew)
ynew10 = f10(xnew)

ynew11 = f11(xnew)
ynew12 = f12(xnew)

ynew13 = f13(xnew)
ynew14 = f14(xnew)

ynew15 = f15(xnew)
ynew16 = f16(xnew)

ynew17 = f17(xnew)
ynew18 = f18(xnew)

ynew19 = f19(xnew)
ynew20 = f20(xnew)

plt.figure()
# plt.figure(figsize=(4,3))
# fig = pylab.figure()

lw = 1
plt.rc('font', size=15)
plt.rc('legend', fontsize=13)

plt.yticks(np.arange(0,105,2))
plt.xticks(np.arange(0,105,2))

plt.plot(np.arange(100,0,-1), color='black',
         lw=lw, label='EER line')
plt.plot(xnew*100, ((ynew1*100)+(ynew2*100))/2.0,
         lw=lw, label='U-net with CLE')
plt.plot(xnew*100, ((ynew3*100)+(ynew4*100))/2.0,
         lw=lw, label='SN-DCR')
plt.plot(xnew*100, ((ynew5*100)+(ynew6*100))/2.0,
         lw=lw, label='U-transformer')
plt.plot(xnew*100, ((ynew7*100)+(ynew8*100))/2.0,
         lw=lw, label='QueryOTR')
plt.plot(xnew*100, ((ynew9*100)+(ynew10*100))/2.0,
         lw=lw, label='CUT')
plt.plot(xnew*100, ((ynew11*100)+(ynew12*100))/2.0,
         lw=lw, label='Harmonization GAN')
plt.plot(xnew*100, ((ynew13*100)+(ynew14*100))/2.0,
         lw=lw, label='Pix2Pix-HD')
plt.plot(xnew*100, ((ynew15*100)+(ynew16*100))/2.0,
         lw=lw, label='CycleGAN')
plt.plot(xnew*100, ((ynew17*100)+(ynew18*100))/2.0,
         lw=lw, label='Pix2Pix')
plt.plot(xnew*100, ((ynew19*100)+(ynew20*100))/2.0,
         lw=lw, label='DION4FR')


plt.xlim([0, 10])
plt.ylim([90, 100])
plt.xlabel('FAR (%)')
plt.ylabel('GAR (%)')
# plt.legend(loc="center right",  frameon=False, bbox_to_anchor=(1.5,0.5))

plt.savefig('EER_HKPU_COMPARE.png')

# fig_leg = plt.figure(figsize=(3,3))
# ax_leg = fig_leg.add_subplot(111)
# # add the legend from the previous axes
# ax_leg.legend(*plt.get_legend_handles_labels(), loc='center')
# # hide the axes frame and the x/y labels
# ax_leg.axis('off')
# fig_leg.savefig('legend.png')


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