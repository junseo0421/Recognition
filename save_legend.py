### 아래의 것 참조 + make_ROC_CURVE_scikitlearn_proposed.py 해서 하기

import pylab
fig = pylab.figure()
figlegend = pylab.figure()
ax = fig.add_subplot(111)
lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10), range(10), pylab.randn(10), range(10), pylab.randn(10)
                , range(10), pylab.randn(10), range(10), pylab.randn(10), range(10), pylab.randn(10), range(10), pylab.randn(10)
                , range(10), pylab.randn(10), range(10), pylab.randn(10))
figlegend.legend(lines, ('U-net with CLE', 'SN-DCR','U-transformer','QueryOTR','CUT','Harmonization GAN','Pix2Pix-HD'
                         ,'CycleGAN','Pix2Pix','DION4FR'), 'center',frameon=False)
fig.show()
figlegend.show()
figlegend.savefig('legend.png')