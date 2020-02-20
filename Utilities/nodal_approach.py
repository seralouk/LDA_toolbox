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


def read_ctrl_nodal(ctrl_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     ctrl_subj: A list with subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     3  network features per node: deg, cc, ne for the CTRL group and concatinate them in one dataframe
    """
    counter = 0
    for s in ctrl_subj:
          X=pd.read_csv(filename.format(s),header=None)
          X=X.values
          np.fill_diagonal(X,0)
          X=X.tolist()
          g = Graph.Weighted_Adjacency(X, mode=ADJ_UNDIRECTED, attr="weight", loops = False)
          N=g.vcount()
          E=g.ecount()
          g.es["label"] = g.es["weight"]
          lst = range(N)
          g.vs["name"] = lst

          d = g.strength(loops=False, weights=g.es["weight"]) 
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])   
          ne = nodal_eff(g)

          d=np.array(d)
          cc=np.array(cc)

          if counter == 0:
            dt1 = pd.DataFrame(columns=['Deg']*d.shape[0] + ['Cc']*d.shape[0] + ['Ne']*d.shape[0] +['ID'])

          all_together = np.concatenate((d,cc,ne), axis = 0)
          all_together= np.append(all_together, 1)
          dt1.loc[s] = list(all_together)
          counter +=1
    return dt1



def read_ep_nodal(ep_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     ep_subj: A list with subject numbers. e.g. for 3 subjects use ep_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'ep_subj_{}'

    Returns:
     3 network features per node: deg, cc, ne for the EP group and concatinate them in one dataframe
    """

    counter = 0
    for s in ep_subj:
          X=pd.read_csv(filename.format(s),header=None)
          X=X.values
          np.fill_diagonal(X,0)
          X=X.tolist()
          g = Graph.Weighted_Adjacency(X, mode=ADJ_UNDIRECTED, attr="weight", loops = False)
          N=g.vcount()
          E=g.ecount()
          g.es["label"] = g.es["weight"]
          lst = range(N)
          g.vs["name"] = lst

          d = g.strength(loops=False, weights=g.es["weight"])
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          ne = nodal_eff(g)

          d=np.array(d)
          cc=np.array(cc)

          if counter == 0:
            dt2 = pd.DataFrame(columns=['Deg']*d.shape[0] + ['Cc']*d.shape[0] + ['Ne']*d.shape[0] +['ID'])
          
          all_together = np.concatenate((d,cc,ne), axis = 0)
          all_together= np.append(all_together, 2)
          dt2.loc[s] = list(all_together)
          counter +=1

    return dt2



def read_iugr_nodal(iugr_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     iugr_subj: A list with subject numbers. e.g. for 3 subjects use iugr_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'iugr_subj_{}'

    Returns:
      3 network features per node: deg, cc, ne for the IUGR group and concatinate them in one dataframe
    """
  
    counter = 0
    for s in iugr_subj:
          X=pd.read_csv(filename.format(s),header=None)
          X=X.values
          np.fill_diagonal(X,0)
          X=X.tolist()
          g = Graph.Weighted_Adjacency(X, mode=ADJ_UNDIRECTED, attr="weight", loops = False)
          N=g.vcount()
          E=g.ecount()
          g.es["label"] = g.es["weight"]
          lst = range(N)
          g.vs["name"] = lst

          d = g.strength(loops=False, weights=g.es["weight"])
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          ne = nodal_eff(g)

          d=np.array(d)
          cc=np.array(cc)         

          if counter == 0:
            dt3 = pd.DataFrame(columns=['Deg']*d.shape[0] + ['Cc']*d.shape[0] + ['Ne']*d.shape[0] +['ID'])
          
          all_together = np.concatenate((d,cc,ne), axis = 0)
          all_together= np.append(all_together, 3)
          dt3.loc[s] = list(all_together)
          counter +=1
          

    return dt3


def read_vav_nodal(vav_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     vav_subj: A list with subject numbers. e.g. for 3 subjects use vav_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'iugr_subj_{}'

    Returns:
      3 network features per node: deg, cc, ne for the IUGR group and concatinate them in one dataframe
    """
  
    counter = 0
    for s in vav_subj:
          X=pd.read_csv(filename.format(s),header=None)
          X=X.values
          np.fill_diagonal(X,0)
          X=X.tolist()
          g = Graph.Weighted_Adjacency(X, mode=ADJ_UNDIRECTED, attr="weight", loops = False)
          N=g.vcount()
          E=g.ecount()
          g.es["label"] = g.es["weight"]
          lst = range(N)
          g.vs["name"] = lst

          d = g.strength(loops=False, weights=g.es["weight"])
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          ne = nodal_eff(g)

          d=np.array(d)
          cc=np.array(cc)         

          if counter == 0:
            dt4 = pd.DataFrame(columns=['Deg']*d.shape[0] + ['Cc']*d.shape[0] + ['Ne']*d.shape[0] +['ID'])
          
          all_together = np.concatenate((d,cc,ne), axis = 0)
          all_together= np.append(all_together, 3)
          dt4.loc[s] = list(all_together)
          counter +=1
          

    return dt4
