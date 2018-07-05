"""
=========================================
Author: Serafeim Loukas, May 2018
=========================================

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

save_results_to = '/Users/miplab/Desktop/LDA_toolbox/Results/'
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)


def get_weights(data ,number_of_features = 3):
    """
    =========================================
    Author: Serafeim Loukas, May 2018
    =========================================

    """
    data = data.iloc[:,1:]
    n_tot_features = data.shape[1] - 1

    X, y = data.iloc[0: , 0:n_tot_features].values, data.iloc[0:,n_tot_features].values
    print("x and y are {} and {}".format(X.shape, y.shape))

    sc = StandardScaler()
    X= sc.fit_transform(X)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X,y)

    df_ctrl_vs_ep= pd.DataFrame({'Features': np.transpose(range(n_tot_features)) , 
                                'Weights': np.transpose(clf.coef_[0,:]).reshape(n_tot_features,)})
    df_ctrl_vs_iugr= pd.DataFrame({'Features': np.transpose(range(n_tot_features)) ,
                                'Weights': np.transpose(clf.coef_[1,:]).reshape(n_tot_features,)})
    df_ep_vs_iugr= pd.DataFrame({'Features': np.transpose(range(n_tot_features)) ,
                                'Weights': np.transpose(clf.coef_[2,:]).reshape(n_tot_features,)})

    Degree = range(0, len(clf.coef_[0]), number_of_features)
    Cl_coef = range(1, len(clf.coef_[0]), number_of_features)
    Nod_eff = range(2, len(clf.coef_[0]), number_of_features)

    ctrl_vs_ep_weights = clf.coef_[0]
    Degree_Weights_ctrl_vs_ep = ctrl_vs_ep_weights[Degree]

    ctrl_vs_iugr_weights = clf.coef_[1]
    Degree_Weights_ctrl_vs_iugr = ctrl_vs_iugr_weights[Degree]

    ep_vs_iugr_weights = clf.coef_[2]
    Degree_Weights_ep_vs_iugr = ep_vs_iugr_weights[Degree]

    return Degree_Weights_ctrl_vs_ep, Degree_Weights_ctrl_vs_iugr, Degree_Weights_ep_vs_iugr

    
    # Degree = range(0,83)
    # Cl_coef = range(83,166)
    # Nod_eff = range(166,249)

    # Degree_Weights_ctrl_vs_ep = df_ctrl_vs_ep.iloc[Degree]
    # Cl_Weights_ctrl_vs_ep = df_ctrl_vs_ep.iloc[Cl_coef]
    # Cl_Weights_ctrl_vs_ep.reset_index(inplace=True)
    # Nod_Weights_ctrl_vs_ep = df_ctrl_vs_ep.iloc[Nod_eff]
    # Nod_Weights_ctrl_vs_ep.reset_index(inplace=True)

    # Degree_Weights_ctrl_vs_iugr = df_ctrl_vs_iugr.iloc[Degree]
    # Cl_Weights_ctrl_vs_iugr = df_ctrl_vs_iugr.iloc[Cl_coef]
    # Cl_Weights_ctrl_vs_iugr.reset_index(inplace=True)
    # Nod_Weights_ctrl_vs_iugr = df_ctrl_vs_iugr.iloc[Nod_eff]
    # Nod_Weights_ctrl_vs_iugr.reset_index(inplace=True)

    # Degree_Weights_ep_vs_iugr = df_ep_vs_iugr.iloc[Degree]
    # Cl_Weights_ep_vs_iugr = df_ep_vs_iugr.iloc[Cl_coef]
    # Cl_Weights_ep_vs_iugr.reset_index(inplace=True)
    # Nod_Weights_ep_vs_iugr = df_ep_vs_iugr.iloc[Nod_eff]
    # Nod_Weights_ep_vs_iugr.reset_index(inplace=True)


    # tmp = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp.iloc[:,4] = np.abs(Degree_Weights_ctrl_vs_ep["weights"])
    # tmp.to_csv('Degree_Weights_ctrl_vs_ep.txt', header=None, index=None, sep=' ', mode='a')

    # tmp2 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp2.iloc[:,4] = np.abs(Cl_Weights_ctrl_vs_ep["weights"])
    # tmp2.to_csv('Cl_Weights_ctrl_vs_ep.txt', header=None, index=None, sep=' ', mode='a')

    # tmp3 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp3.iloc[:,4] = np.abs(Nod_Weights_ctrl_vs_ep["weights"])
    # tmp3.to_csv('Nod_Weights_ctrl_vs_ep.txt', header=None, index=None, sep=' ', mode='a')


    # tmp4 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp4.iloc[:,4] = np.abs(Degree_Weights_ctrl_vs_iugr["weights"])
    # tmp4.to_csv('Degree_Weights_ctrl_vs_iugr.txt', header=None, index=None, sep=' ', mode='a')

    # tmp5 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp5.iloc[:,4] = np.abs(Cl_Weights_ctrl_vs_iugr["weights"])
    # tmp5.to_csv('Cl_Weights_ctrl_vs_iugr.txt', header=None, index=None, sep=' ', mode='a')

    # tmp6 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp6.iloc[:,4] = np.abs(Nod_Weights_ctrl_vs_iugr["weights"])
    # tmp6.to_csv('Nod_Weights_ctrl_vs_iugr.txt', header=None, index=None, sep=' ', mode='a')


    # tmp7 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp7.iloc[:,4] = np.abs(Degree_Weights_ep_vs_iugr["weights"])
    # tmp7.to_csv('Degree_Weights_ep_vs_iugr.txt', header=None, index=None, sep=' ', mode='a')

    # tmp8 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp8.iloc[:,4] = np.abs(Cl_Weights_ep_vs_iugr["weights"])
    # tmp8.to_csv('Cl_Weights_ep_vs_iugr.txt', header=None, index=None, sep=' ', mode='a')

    # tmp9 = pd.read_csv('clustering1.node', sep=" ", header=None)
    # tmp9.iloc[:,4] = np.abs(Nod_Weights_ep_vs_iugr["weights"])
    # tmp9.to_csv('Nod_Weights_ep_vs_iugr.txt', header=None, index=None, sep=' ', mode='a')


