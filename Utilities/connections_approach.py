"""
=========================================
Author: Serafeim Loukas, May 2018
=========================================
"""
from numpy import *
from igraph import *
import string
from nodal_efficiency import nodal_eff
import time 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def read_ctrl_connections(ctrl_subj, filename, number_of_brain_regions_in_connectomes):
  """
  This function loads the connectomes of the CTRL group adding 0s in the diagonal.
    
  Created by: Loukas Serafeim, Nov 2017

  Args:
   ctrl_subj: A list with subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
   filename: This is the filename of the subjects. e.g. filename = 'ctrl_subj_{}'

  Returns:
   The connectomes of the CTRL stored in a tensor X
  """
  n_of_reg = number_of_brain_regions_in_connectomes
  X = np.zeros((n_of_reg, n_of_reg, len(ctrl_subj)))

  for s, ss in enumerate(ctrl_subj):
    XX=pd.read_csv(filename.format(ss),header=None)
    for se in range(X.shape[0]):
      XX.iloc[se,se] = 0.0
    
    #np.fill_diagonal(XX.values, 0)
    
    X[:,:,s-1] = XX

  return X



def read_ep_connections(ep_subj, filename, number_of_brain_regions_in_connectomes):
  """
  This function loads the connectomes of the EP group adding 0s in the diagonal.
    
  Created by: Loukas Serafeim, Nov 2017

  Args:
   ep_subj: A list with subject numbers. e.g. for 3 subjects use ep_subj= [1, 2, 3]
   filename: This is the filename of the subjects. e.g. filename = 'ep_subj_{}'

  Returns:
   The connectomes of the EP stored in a tensor X
  """
  n_of_reg = number_of_brain_regions_in_connectomes
  X = np.zeros((n_of_reg, n_of_reg, len(ep_subj)))

  for s, ss in enumerate(ep_subj):
    XX=pd.read_csv(filename.format(ss),header=None)
    for se in range(X.shape[0]):
      XX.iloc[se,se] = 0.0
    
    #np.fill_diagonal(XX.values, 0)
    
    X[:,:,s-1] = XX

  return X



def read_iugr_connections(iugr_subj, filename, number_of_brain_regions_in_connectomes):
  """
  This function loads the connectomes of the IUGR group adding 0s in the diagonal.
    
  Created by: Loukas Serafeim, Nov 2017

  Args:
   iugr_subj: A list with subject numbers. e.g. for 3 subjects use iugr_subj= [1, 2, 3]
   filename: This is the filename of the subjects. e.g. filename = 'iugr_subj_{}'

  Returns:
   The connectomes of the IUGR stored in a tensor X
  """
  n_of_reg = number_of_brain_regions_in_connectomes
  X = np.zeros((n_of_reg, n_of_reg, len(iugr_subj)))

  for s, ss in enumerate(iugr_subj):
    XX=pd.read_csv(filename.format(ss),header=None)
    for se in range(X.shape[0]):
      XX.iloc[se,se] = 0.0
    
    #np.fill_diagonal(XX.values, 0)
    
    X[:,:,s-1] = XX

  return X
