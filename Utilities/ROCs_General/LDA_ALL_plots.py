import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show,plot,draw
from matplotlib import style
style.use('ggplot')

plt.figure(figsize=(11,9))

labels = ('Global', 'FG', 'LE')
x_cords1 = np.arange(len(labels))
heights1 = [0.8269, 0.8653, 0.8461]
plt.subplot(1,3,1)
plt.ylim([0.5,1])
#plt.ylim([0.5, 1])
plt.bar([0], [0.8269], align='center', width=0.70, color=['r'],label='Global',alpha=0.65)
plt.bar([1],[0.8653],align='center', width=0.70, color=['g'],label='FG',alpha=0.65)
plt.bar([2],[0.8461],align='center', width=0.70, color=['b'],label='LE',alpha=0.65)
#plt.bar(x_cords, heights, align='center',width=0.75,color=['r','b','g'], alpha=0.55)
plt.tick_params(axis='x',labelsize=12)
plt.xticks(x_cords1, labels)
plt.ylabel('Accuracy LOOCV')
#plt.xlabel('Algorithms')
plt.title('3-groups',fontsize=15)
plt.legend(loc=4,prop={'size':12})

labels2= ('Global', 'Fast_Greedy', 'Leading_Eigen')
x_cords2=np.arange(len(labels))
heights2= [0.9230, 0.9807, 1]
plt.subplot(1,3,2)
plt.ylim([0.5,1])
#plt.ylim([0.5, 1])
plt.bar([0], [0.9230], align='center', width=0.70, color=['r'],label='Global',alpha=0.65)
plt.bar([1],[0.9807],align='center', width=0.70, color=['g'],label='FG',alpha=0.65)
plt.bar([2],[1],align='center', width=0.70, color=['b'],label='LE',alpha=0.65)
#plt.bar(x_cords, heights, align='center',width=0.75,color=['r','b','g'], alpha=0.55)
plt.tick_params(axis='x',labelsize=12)
plt.xticks(x_cords2, labels)
plt.ylabel('Accuracy LOOCV')
#plt.xlabel('Algorithms')
plt.title('MP/CTRL - PM',fontsize=15)
plt.legend(loc=4,prop={'size':12})


x_cords3=np.arange(len(labels))
heights3= [0.8863, 0.8863, 0.7954]
plt.subplot(1,3,3)
plt.ylim([0.5,1])
#plt.ylim([0.5, 1])
plt.bar([0], [0.88], align='center', width=0.70, color=['r'],label='Global',alpha=0.65)
plt.bar([1],[0.8863],align='center', width=0.70, color=['g'],label='FG',alpha=0.65)
plt.bar([2],[0.79],align='center', width=0.70, color=['b'],label='LE',alpha=0.65)
#plt.bar(x_cords, heights, align='center',width=0.75,color=['r','b','g'], alpha=0.55)
plt.tick_params(axis='x',labelsize=12)
plt.xticks(x_cords3, labels)
plt.ylabel('Accuracy LOOCV')
#plt.xlabel('Algorithms')
plt.title('EP - IUGR',fontsize=15)
plt.legend(loc=4,prop={'size':12})

plt.suptitle('LDA Classification Accuracy for Module and Global levels',fontsize=25)
#plt.tight_layout()
plt.savefig("LDA_results.png", dpi = 300)
plt.show()
