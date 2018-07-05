"""
=========================================
Author: Serafeim Loukas, May 2018
=========================================

"""
from numpy import *
from igraph import *
seterr(divide='ignore')


def nodal_eff(g):
    """
    This function calculates the nodal efficiency of a weighted graph object.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
     g: A igraph Graph() object.

    Returns:
     The nodal efficiency of each node of the graph
    """
    
    g.es["weight"] = [1.0 / x for x in g.es["weight"]]
    sp = g.shortest_paths_dijkstra(weights=g.es["weight"])
    sp = asarray(sp)
    temp =1/sp
    fill_diagonal(temp,0)
    N=temp.shape[0]
    ne= ( 1.0 / (N-1)) * apply_along_axis(sum,0,temp)

    return ne