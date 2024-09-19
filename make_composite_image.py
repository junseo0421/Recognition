## 논문에 실을 composite 이미지 만들기
from utility.datasetutil import *
import skimage.io as iio

## img_name1 enrolled / img_name2 target
img_name1='Datasets/GRADCAM_INPUT1/0_98_6_f1.bmp'
img_name2='Datasets/GRADCAM_INPUT2/left.bmp'

img1 = iio.imread(img_name1)
img2 = iio.imread(img_name2)
# 3 채널 #height 기준 concatenation)
# img3_1 = skiT.resize(img1, (224, 224))
# iio.imsave('0_77_5_f2.jpg',(img3_1*255).astype(np.ubyte))
# img3_2 = skiT.resize(img2, (112, 224))
# img3 = np.reshape(np.concatenate([img3_1, img3_2], axis=0), newshape=(224, 224))

composite_img = make_composite_image(img1,img2)
# iio.imsave('enroll_0_1_1_f1.bmp',img1)
# iio.imsave('imposter_2_54_1_f1.bmp',img2)
# iio.imsave('concatenated_en_im.bmp',(img3*255).astype(np.ubyte))
iio.imsave('Gg_composite_left_0_98_6_f1.bmp',(composite_img*255).astype(np.ubyte))

