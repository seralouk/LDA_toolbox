from numpy import *
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
plt.style.use('ggplot')
lw = 2

save_to = '/Users/miplab/Desktop/LDA_toolbox/Results/'
data_path = '/Users/miplab/Downloads/LDA_toolbox_new/Results/'


def plot_ROCs():
#########################################################
################## Apply LDA on Global data #############
#########################################################

    os.chdir(data_path)
    # Get features and target variable for all the 3 groups
    data = pd.read_csv('global_network_features.csv')
    data = data.iloc[:,1:]
    k=data.shape[1]-1
    X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values

    loo = LeaveOneOut()
    pipe= Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())])
    classes = np.unique(y)
    y_true = label_binarize(y, classes=classes)
    n_classes = y_true.shape[1]
    ypre1 = cross_val_predict(pipe, X, y, cv=loo ,method='predict_proba')

    plt.figure(1, figsize=(14,8))
    plt.subplot(231)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], ypre1[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    names = ['Ctrl','EP','IUGR']
    colors = cycle(['deepskyblue', 'orangered', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC for Global approach', fontsize= 10)
    plt.legend(loc="lower right",prop={'size':9})
    plt.grid(True)
    #plt.show()
    del(X,y)


    ## Second way to do the same thing
    # skplt.metrics.plot_roc_curve(y, ypre1, title='ROC Curves', curves=('each_class'), cmap='nipy_spectral', figsize=(8,8) ,title_fontsize='large')
    # plt.show()



    #########################################################
    ######### Apply LDA on Deep fast greedy data ############
    #########################################################

    # Get features and target variable for all the 3 groups
    data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
    data = data.iloc[:,1:]
    k=data.shape[1]-1
    X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values

    loo = LeaveOneOut()
    pipe= Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())])
    classes = np.unique(y)
    y_true = label_binarize(y, classes=classes)

    n_classes = y_true.shape[1]
    ypre2 = cross_val_predict(pipe, X, y, cv=loo ,method='predict_proba')

    plt.subplot(233)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], ypre2[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['deepskyblue', 'orangered', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC for Modular (FG) approach', fontsize= 10)
    plt.legend(loc="lower right",prop={'size': 9})
    plt.grid(True)
    #plt.show()

    del(X,y)
    #########################################################
    ###### Apply LDA on Deep leading eigen data #############
    #########################################################

    # Get features and target variable for all the 3 groups
    data = pd.read_csv('deep_leading_eigenvector_comm_network_features.csv')
    data = data.iloc[:,1:]
    X, y = data.iloc[0:, 0:18].values, data.iloc[0:,18].values

    loo = LeaveOneOut()
    pipe= Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())])
    classes = np.unique(y)
    y_true = label_binarize(y, classes=classes)

    n_classes = y_true.shape[1]
    ypre3 = cross_val_predict(pipe, X, y, cv=loo ,method='predict_proba')


    plt.subplot(235)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], ypre3[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['deepskyblue', 'orangered', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC for Modular (LE) approach', fontsize= 10)
    plt.legend(loc="lower right",prop={'size': 9})
    plt.grid(True)
    #plt.show()

    del(X,y)


    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.suptitle('Receiver operating characteristic (ROC) Curves', fontsize=18)
    plt.savefig(save_to + 'ROCs_LDA.png' ,dpi = 450)
    

    return plt