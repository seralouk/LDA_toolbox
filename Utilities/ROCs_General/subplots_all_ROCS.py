from numpy import *
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.neural_network import MLPClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from itertools import cycle
from sklearn import preprocessing

plt.style.use('ggplot')

save_to = '/Users/miplab/Desktop/'
data_path = '/Users/miplab/Desktop/LDA_toolbox/Results/'

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(16,8))
params = {'xtick.labelsize':7, 'ytick.labelsize':7}
plt.rcParams.update(params)



exclude_class_label = 1
#########################################################
################## Global data #############
#########################################################
os.chdir(data_path)
# Get features and target variable for all the 3 groups
data = pd.read_csv('global_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
#y = label_binarize(y, classes=[0,1])
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
plt.figure(1, figsize=(14,8))
#plt.subplot(231)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.sca(ax[0,0])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='coral',
         label=r'Mean ROC-Global approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='coral', alpha=.4)
del(X,y, pipe)

#########################################################
#########       Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='dodgerblue',
         label=r'Mean ROC-Module (FG) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.6)
del(X,y, pipe)

#########################################################
######                  Nodal data          #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('nodal_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='g',
         label=r'Mean ROC-Nodal approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='g', alpha=.2)
del(X,y)

#########################################################
######                Connections data      #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='grey',
         label=r'Mean ROC-Connections approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
del(X,y)

#########################################################
######                Connections MLP      #############
#########################################################


# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(2246, ),solver="sgd", activation='relu',random_state=0))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='yellow',
         label=r'Mean ROC-Connections approach (MLP) (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='yellow', alpha=.1)
del(X,y)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
class_names_all = ['MP/CTRL', 'EP', 'IUGR']
plt.title('ROC excluded class {}'.format(class_names_all[exclude_class_label-1]), fontsize= 10)
plt.legend(loc="lower right",prop={'size':7})
plt.grid(True)
#plt.axes().set_aspect('equal', 'datalim')
plt.subplots_adjust(left = 0.07, bottom =0.09, wspace=0.14, hspace=0.21, right = 0.95, top =0.89)
#plt.suptitle('Cross-validated Receiver operating characteristic (ROC) Curves', fontsize=10)
#plt.savefig(save_to + 'ROCs_ALL_excluded_{}.png'.format(exclude_class_label) ,dpi = 450)




#########################################################
################## LE data #############
#########################################################
os.chdir(data_path)
# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_leading_eigenvector_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
#y = label_binarize(y, key=[0,1])
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
plt.figure(1, figsize=(14,8))
#plt.subplot(231)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.sca(ax[1,0])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='coral',
         label=r'Mean ROC-Modular (LE) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='coral', alpha=.4)
del(X,y, pipe)

#########################################################
#########       Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='dodgerblue',
         label=r'Mean ROC-Modular (FG) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.3)
del(X,y, pipe)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
class_names_all = ['MP/CTRL', 'EP', 'IUGR']
#plt.title('ROC excluded class {}'.format(class_names_all[exclude_class_label-1]), fontsize= 10)
plt.legend(loc="lower right",prop={'size':7})
plt.grid(True)
#plt.axes().set_aspect('equal', 'datalim')
plt.subplots_adjust(left = 0.07, bottom =0.09, wspace=0.14, hspace=0.21, right = 0.95, top =0.89)
plt.suptitle('Cross-validated Receiver operating characteristic (ROC) Curves', fontsize=14)
#plt.savefig(save_to + 'ROCs_DECOMP_excluded_{}.png'.format(exclude_class_label) ,dpi = 450)


#######################################################################################################
######################################################################################################


del(exclude_class_label)
exclude_class_label = 2
#########################################################
################## Global data #############
#########################################################
os.chdir(data_path)
# Get features and target variable for all the 3 groups
data = pd.read_csv('global_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
#y = label_binarize(y, classes=[0,1])
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
plt.figure(1, figsize=(14,8))
#plt.subplot(231)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.sca(ax[0,1])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='coral',
         label=r'Mean ROC-Global approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='coral', alpha=.4)
del(X,y, pipe)

#########################################################
#########       Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='dodgerblue',
         label=r'Mean ROC-Module (FG) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=0.3)
del(X,y, pipe)

#########################################################
######                  Nodal data          #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('nodal_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='g',
         label=r'Mean ROC-Nodal approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='g', alpha=.2)
del(X,y)

#########################################################
######                Connections data      #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='grey',
         label=r'Mean ROC-Connections approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
del(X,y)

#########################################################
######                Connections MLP      #############
#########################################################


# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(2246, ),solver="sgd", activation='relu',random_state=0))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='yellow',
         label=r'Mean ROC-Connections approach (MLP) (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='yellow', alpha=.1)
del(X,y)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
class_names_all = ['MP/CTRL', 'EP', 'IUGR']
plt.title('ROC excluded class {}'.format(class_names_all[exclude_class_label-1]), fontsize= 10)
plt.legend(loc="lower right",prop={'size':7})
plt.grid(True)
#plt.axes().set_aspect('equal', 'datalim')
plt.subplots_adjust(wspace=0.1, hspace=0.25)
#plt.suptitle('Cross-validated Receiver operating characteristic (ROC) Curves', fontsize=10)
#plt.savefig(save_to + 'ROCs_ALL_excluded_{}.png'.format(exclude_class_label) ,dpi = 450)




#########################################################
################## LE data #############
#########################################################
os.chdir(data_path)
# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_leading_eigenvector_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
#y = label_binarize(y, key=[0,1])
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
plt.figure(1, figsize=(14,8))
#plt.subplot(231)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.sca(ax[1,1])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='coral',
         label=r'Mean ROC-Modular (LE) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='coral', alpha=.4)
del(X,y, pipe)

#########################################################
#########       Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr-0.005, mean_tpr-0.005, color='dodgerblue',
         label=r'Mean ROC-Modular (FG) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.3)
del(X,y, pipe)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
class_names_all = ['MP/CTRL', 'EP', 'IUGR']
#plt.title('ROC excluded class {}'.format(class_names_all[exclude_class_label-1]), fontsize= 10)
plt.legend(loc="lower right",prop={'size':7})
plt.grid(True)
#plt.axes().set_aspect('equal', 'datalim')
plt.subplots_adjust(left = 0.07, bottom =0.09, wspace=0.14, hspace=0.21, right = 0.95, top =0.89)
#plt.suptitle('Cross-validated Receiver operating characteristic (ROC) Curves', fontsize=14)
#plt.savefig(save_to + 'ROCs_DECOMP_excluded_{}.png'.format(exclude_class_label) ,dpi = 450)


#######################################################################################################
######################################################################################################


del(exclude_class_label)
exclude_class_label = 3
#########################################################
################## Global data #############
#########################################################
os.chdir(data_path)
# Get features and target variable for all the 3 groups
data = pd.read_csv('global_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
#y = label_binarize(y, classes=[0,1])
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
plt.figure(1, figsize=(14,8))
#plt.subplot(231)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.sca(ax[0,2])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='coral',
         label=r'Mean ROC-Global approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='coral', alpha=.4)
del(X,y, pipe)

#########################################################
#########       Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='dodgerblue',
         label=r'Mean ROC-Module (FG) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.6)
del(X,y, pipe)

#########################################################
######                  Nodal data          #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('nodal_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='g',
         label=r'Mean ROC-Nodal approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='g', alpha=.2)
del(X,y)

#########################################################
######                Connections data      #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='grey',
         label=r'Mean ROC-Connections approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
del(X,y)

#########################################################
######                Connections MLP      #############
#########################################################


# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(2246, ),solver="sgd", activation='relu',random_state=0))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='yellow',
         label=r'Mean ROC-Connections approach (MLP) (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.80)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='yellow', alpha=.1)
del(X,y)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
class_names_all = ['MP/CTRL', 'EP', 'IUGR']
plt.title('ROC excluded class {}'.format(class_names_all[exclude_class_label-1]), fontsize= 10)
plt.legend(loc="lower right",prop={'size':7})
plt.grid(True)
#plt.axes().set_aspect('equal', 'datalim')
plt.subplots_adjust(left = 0.07, bottom =0.09, wspace=0.14, hspace=0.21, right = 0.95, top =0.89)
#plt.suptitle('Cross-validated Receiver operating characteristic (ROC) Curves', fontsize=10)
#plt.savefig(save_to + 'ROCs_ALL_excluded_{}.png'.format(exclude_class_label) ,dpi = 450)




#########################################################
################## LE data #############
#########################################################
os.chdir(data_path)
# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_leading_eigenvector_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
#y = label_binarize(y, key=[0,1])
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
plt.figure(1, figsize=(14,8))
#plt.subplot(231)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.sca(ax[1,2])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='coral',
         label=r'Mean ROC-Modular (LE) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='coral', alpha=.4)
del(X,y, pipe)

#########################################################
#########       Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
X, y = X[y != exclude_class_label], y[y != exclude_class_label]
y = preprocessing.LabelBinarizer().fit_transform(y)
cv = KFold(n_splits = 5, shuffle = True, random_state= 0)
pipe= Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear',probability=True,random_state = 1))])
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = pipe.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance level', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr-0.005, mean_tpr-0.005, color='dodgerblue',
         label=r'Mean ROC-Modular (FG) approach (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.95)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=0.3)
del(X,y, pipe)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
class_names_all = ['MP/CTRL', 'EP', 'IUGR']
#plt.title('ROC excluded class {}'.format(class_names_all[exclude_class_label-1]), fontsize= 10)
plt.legend(loc="lower right",prop={'size':7})
plt.grid(True)
#plt.axes().set_aspect('equal', 'datalim')
plt.subplots_adjust(left = 0.07, bottom =0.09, wspace=0.14, hspace=0.21, right = 0.95, top =0.89)
plt.suptitle('Cross-validated Receiver operating characteristic (ROC) Curves', fontsize=14)
plt.savefig(save_to + 'ROCs.png',dpi = 300)
plt.show()
