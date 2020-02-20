"""
=========================================
Author: Serafeim Loukas, May 2019
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


def read_ctrl(ctrl_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     ctrl_subj: A list with subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     3 Global network features: deg, cc, ne for the CTRL group and concatinate them in one dataframe
    """

    dt1 = pd.DataFrame(columns=['Deg', 'Cc', 'Ne','ID'])
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
          av_d = average(d)  
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          av_cc = average(cc)    
          ne = nodal_eff(g)
          av_ne = average(ne)
          
          dt1.loc[s] =[av_d,av_cc,av_ne, 1]

    return dt1



def read_ep(ep_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     ep_subj: A list with subject numbers. e.g. for 3 subjects use ep_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'ep_subj_{}'

    Returns:
     3 Global network features: deg, cc, ne for the EP group and concatinate them in one dataframe
    """

    dt2 = pd.DataFrame(columns=['Deg', 'Cc', 'Ne','ID'])
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
          av_d = average(d)  
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          av_cc = average(cc)    
          ne = nodal_eff(g)
          av_ne = average(ne)
          
          dt2.loc[s] =[av_d,av_cc,av_ne, 2]

    return dt2



def read_iugr(iugr_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     iugr_subj: A list with subject numbers. e.g. for 3 subjects use iugr_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'iugr_subj_{}'

    Returns:
      3 Global network features: deg, cc, ne for the IUGR group and concatinate them in one dataframe
    """
  
    dt3 = pd.DataFrame(columns=['Deg', 'Cc', 'Ne','ID'])
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
          av_d = average(d)  
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          av_cc = average(cc)     
          ne = nodal_eff(g)
          av_ne = average(ne)
          
          dt3.loc[s] =[av_d,av_cc,av_ne, 3]

    return dt3

def read_vav(vav_subj, filename):
    """
    This function creates an empty dataframe, builds the brain graphs,
    calculates the network features and writes them in the dataframe.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     vav_subj: A list with subject numbers. e.g. for 3 subjects use vav_subj= [1, 2, 3]
     filename: This is the filename of the subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     3 Global network features: deg, cc, ne for the VAV group and concatinate them in one dataframe
    """

    dt4 = pd.DataFrame(columns=['Deg', 'Cc', 'Ne','ID'])
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
          av_d = average(d)  
          cc = g.transitivity_local_undirected(vertices=None,mode="zero", weights=g.es["weight"])
          av_cc = average(cc)    
          ne = nodal_eff(g)
          av_ne = average(ne)
          
          dt4.loc[s] =[av_d,av_cc,av_ne, 4]

    return dt4


def mean_ctrl(Subjects, N_regions, filename):
    """
    This function calculates the Mean CTRL network. The decomposition of this network
    will be used as prior information for the decomposition of the other groups.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     Subjects: A list with the CTRL subject numbers. e.g. for 4 CTRL subjects use Subjects= [1, 2, 3, 4]
     N_regions: The number of nodes (brain regions) in the connectomes.
     filename: This is the filename of the CTRL subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
      The mean ctrl network in a pandas dataframe
    """
    X = np.zeros((N_regions, N_regions, len(Subjects)))
    for idx, s in enumerate(Subjects):
        XX=pd.read_csv(filename.format(s),header=None)
        for se in range(X.shape[0]):
          XX.iloc[se,se] = 0.0
        X[:,:,idx] = XX
    Xout = np.sum(X, axis=2) / float(len(Subjects))
    Xout = pd.DataFrame(Xout)

    return Xout

    # x1 = pd.read_csv('ctrl_FA_1.csv',header=None)
    # x2 = pd.read_csv('ctrl_FA_2.csv',header=None) 
    # x3 = pd.read_csv('ctrl_FA_3.csv',header=None)
    # x4 = pd.read_csv('ctrl_FA_4.csv',header=None) 
    # x5 = pd.read_csv('ctrl_FA_5.csv',header=None)
    # x6 = pd.read_csv('ctrl_FA_6.csv',header=None) 
    # x7 = pd.read_csv('ctrl_FA_7.csv',header=None)
    # x8 = pd.read_csv('ctrl_FA_8.csv',header=None) 
    # X = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
    # X = X / float(8)
    # return X


  
def decompose_ctrl_fg(X, ctrl_subj, filename, verbose=0):
    """
    This function decomposes the CTRL subjects using the fast greedy algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     ctrl_subj: A list with the CTRL subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
     filename: This is the filename of the CTRL subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    X=X.values 
    np.fill_diagonal(X,0)                                            
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X, mode=ADJ_UNDIRECTED, attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_fastgreedy(weights = g.es["weight"])
    clusters = comms.as_clustering()
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')

    dt_ctrl_fg = pd.DataFrame(columns = c)

    for j in ctrl_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(1)
      dt_ctrl_fg.loc[j] = lists

    return dt_ctrl_fg, clusters



def decompose_vav_fg(X, ctrl_subj, filename, verbose=0):
    """
    This function decomposes the CTRL subjects using the fast greedy algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     ctrl_subj: A list with the CTRL subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
     filename: This is the filename of the CTRL subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    X=X.values 
    np.fill_diagonal(X,0)                                            
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X, mode=ADJ_UNDIRECTED, attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_fastgreedy(weights = g.es["weight"])
    clusters = comms.as_clustering()
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')

    dt_ctrl_fg = pd.DataFrame(columns = c)

    for j in ctrl_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(4)
      dt_ctrl_fg.loc[j] = lists

    return dt_ctrl_fg, clusters



def decompose_ep_fg(X, ep_subj, filename, verbose=0):
    """
    This function decomposes the EP subjects using the fast greedy algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     ep_subj: A list with the EP subject numbers. e.g. for 3 subjects use ep_subj= [1, 2, 3]
     filename: This is the filename of the EP subjects. e.g. filename = 'ep_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    # c =['deg', 'cc','ne']*7
    # c.append('id')
    # dt_ep_fg = pd.DataFrame(columns = c)
    X=X.values
    np.fill_diagonal(X,0)                                          
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X,mode=ADJ_UNDIRECTED,attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_fastgreedy(weights = g.es["weight"])
    clusters = comms.as_clustering()
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')
    dt_ep_fg = pd.DataFrame(columns = c)

    for j in ep_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(2)
      dt_ep_fg.loc[j] = lists

    return dt_ep_fg, clusters



def decompose_iugr_fg(X, iugr_subj, filename, verbose=0):
    """
    This function decomposes the IUGR subjects using the fast greedy algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     iugr_subj: A list with the IUGR subject numbers. e.g. for 3 subjects use iugr_subj= [1, 2, 3]
     filename: This is the filename of the IUGR subjects. e.g. filename = 'iugr_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    # c =['deg', 'cc','ne']*7
    # c.append('id')
    # dt_iugr_fg = pd.DataFrame(columns = c)
    X=X.values
    np.fill_diagonal(X,0)                                            
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X,mode=ADJ_UNDIRECTED,attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_fastgreedy(weights = g.es["weight"])
    clusters = comms.as_clustering()
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')
    dt_iugr_fg = pd.DataFrame(columns = c)

    for j in iugr_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(3)
      dt_iugr_fg.loc[j] = lists

    return dt_iugr_fg, clusters


  
def decompose_ctrl_lev(X, ctrl_subj, filename, verbose=0):
    """
    This function decomposes the CTRL subjects using the leading eigenvector algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     ctrl_subj: A list with the CTRL subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
     filename: This is the filename of the CTRL subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    # c =['deg', 'cc','ne']*6
    # c.append('id')
    # dt_ctrl_lev = pd.DataFrame(columns = c)
    X=X.values
    np.fill_diagonal(X,0)                                             
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X,mode=ADJ_UNDIRECTED,attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_leading_eigenvector(weights = g.es["weight"])
    clusters = comms
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')
    dt_ctrl_lev = pd.DataFrame(columns = c)

    for j in ctrl_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(1)
      dt_ctrl_lev.loc[j] = lists

    return dt_ctrl_lev, clusters


def decompose_vav_lev(X, ctrl_subj, filename, verbose=0):
    """
    This function decomposes the CTRL subjects using the leading eigenvector algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     ctrl_subj: A list with the CTRL subject numbers. e.g. for 3 subjects use ctrl_subj= [1, 2, 3]
     filename: This is the filename of the CTRL subjects. e.g. filename = 'ctrl_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    # c =['deg', 'cc','ne']*6
    # c.append('id')
    # dt_ctrl_lev = pd.DataFrame(columns = c)
    X=X.values
    np.fill_diagonal(X,0)                                             
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X,mode=ADJ_UNDIRECTED,attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_leading_eigenvector(weights = g.es["weight"])
    clusters = comms
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')
    dt_ctrl_lev = pd.DataFrame(columns = c)

    for j in ctrl_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(4)
      dt_ctrl_lev.loc[j] = lists

    return dt_ctrl_lev, clusters


def decompose_ep_lev(X, ep_subj, filename, verbose=0):
    """
    This function decomposes the EP subjects using the leading eigenvector algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     ep_subj: A list with the EP subject numbers. e.g. for 3 subjects use ep_subj= [1, 2, 3]
     filename: This is the filename of the EP subjects. e.g. filename = 'ep_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    # c =['deg', 'cc','ne']*6
    # c.append('id')
    # dt_ep_lev = pd.DataFrame(columns = c)
    X=X.values
    np.fill_diagonal(X,0)                                             
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X,mode=ADJ_UNDIRECTED,attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_leading_eigenvector(weights = g.es["weight"])
    clusters = comms
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')
    dt_ep_lev = pd.DataFrame(columns = c)

    for j in ep_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(2)
      dt_ep_lev.loc[j] = lists

    return dt_ep_lev, clusters



def decompose_iugr_lev(X, iugr_subj, filename, verbose=0):
    """
    This function decomposes the IUGR subjects using the leading eigenvector algorithm
    based on the prior information.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     X: The MEAN ctrl network
     iugr_subj: A list with the IUGR subject numbers. e.g. for 3 subjects use iugr_subj= [1, 2, 3]
     filename: This is the filename of the IUGR subjects. e.g. filename = 'iugr_subj_{}'

    Returns:
     The average degree, clustering coefficient and nodal efficiency of each community
     (3*comms features)
    """

    # c =['deg', 'cc','ne']*6
    # c.append('id')
    # dt_iugr_lev = pd.DataFrame(columns = c)
    X=X.values
    np.fill_diagonal(X,0)                                            
    X=X.tolist()

    g = Graph.Weighted_Adjacency(X,mode=ADJ_UNDIRECTED,attr="weight", loops = False)  
    N = g.vcount()
    E = g.ecount()
    g.es["label"] = g.es["weight"]
    g.vs["name"] = range(N)
   
    comms = g.community_leading_eigenvector(weights = g.es["weight"])
    clusters = comms
    if verbose:
      print("\nafter calculating the average control matrix the comms are:  ")
      print(clusters)

    #c =['deg', 'cc','ne'] * len(clusters)
    c = [item for sublist in [['deg_{}'.format(s),'cc_{}'.format(s),'ne_{}'.format(s)] for s in range(len(clusters))] for item in sublist]
    c.append('id')
    dt_iugr_lev = pd.DataFrame(columns = c)

    for j in iugr_subj:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []

      for i in range(len(clusters)):
        sub = gc.induced_subgraph(clusters[i])
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      lists.append(3)
      dt_iugr_lev.loc[j] = lists

    return dt_iugr_lev, clusters




def decompose_lobes(grouping, subj_numbers, filename, verbose=0):

    c = [item for sublist in [['deg_{}_lob'.format(s),'cc_{}_lob'.format(s),'ne_{}_lob'.format(s)] for s in range(len(np.unique(grouping).tolist()))] for item in sublist]
    c.append('id')
    dt_lobe = pd.DataFrame(columns = c)

    for j in subj_numbers:
      Xc = pd.read_csv(filename.format(j), header=None)
      Xc = Xc.values
      np.fill_diagonal(Xc,0)
      Xc = Xc.tolist()
      gc = Graph.Weighted_Adjacency(Xc, mode=ADJ_UNDIRECTED,attr="weight", loops=False)
      Nc = gc.vcount()
      Ec = gc.ecount()
      gc.es["label"] = gc.es["weight"]
      lst_c = range(Nc)
      gc.vs["name"] = lst_c

      lists = []
      for i in range(len(np.unique(grouping))):
        tmp_cluster = np.where(grouping==i)[0].tolist()
        #print(tmp_cluster)
        sub = gc.induced_subgraph(tmp_cluster)
        av_d = average(sub.strength(loops=False,weights=sub.es["weight"]))
        av_cc = average(sub.transitivity_local_undirected(vertices=None,mode="zero", weights=sub.es["weight"]))
        av_ne =average(nodal_eff(sub))

        lists.extend([av_d,av_cc,av_ne])
      if filename == 'ctrl_FA_{}.csv': 
          lists.append(1)
      elif filename == 'ep_FA_{}.csv': 
          lists.append(2)
      elif filename == 'iugr_FA_{}.csv': 
          lists.append(3)
      elif filename == 'ctrl_VAV_{}.csv': 
          lists.append(4)
      dt_lobe.loc[j] = lists

    return dt_lobe, grouping

