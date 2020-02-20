"""
=========================================
Author: Serafeim Loukas, May 2019
=========================================

"""
%load_ext autoreload
%autoreload 2

import os
import sys
import cairocffi as cairo
from colorama import Fore, Back, Style

# Define the paths to utilities, data and results directories.
path_to_utilities = '/Users/loukas/Desktop/LDA_toolbox_new/Utilities/'
path_to_data = '/Users/loukas/Desktop/LDA_toolbox_new/Dataset/'
save_results_to = '/Users/loukas/Desktop/LDA_toolbox_new/Results/'
sys.path.append(path_to_utilities)
sys.path.append(path_to_data)
# Get rid of warnings
def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

################################################################
############- Define global variables and directories -#########
################################################################

# Define the number of regions(nodes) in the connectomes
number_of_brain_regions_in_connectomes = 84
verbose = 1 # 0 -> do not print in console, otherwise print

# Define the finelames as they appear in "path_to_data" directory
control_group_filename = 'ctrl_FA_{}.csv'
ep_group_filename = 'ep_FA_{}.csv'
iugr_group_filename = 'iugr_FA_{}.csv'
vav_group_filename = 'ctrl_VAV_{}.csv'

# os.chdir(path_to_utilities)
from plot_comms import plot_communities
from decomposition_functions import *
from nodal_efficiency import nodal_eff
from LDA_functions import call_LDA_LOOCV
#from SVM_functions import call_SVM_LOOCV
#from MLP_functions import call_MLP_LOOCV
#from plot_results import plot_values_global_modular, plot_values_nodal_connections
from lda_coefficients import get_weights
from nodal_approach import *
from connections_approach import *
from ROC_final import plot_ROCs, plot_ROCs_decomposition
# os.chdir(go_back)

if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

os.chdir(path_to_data)

ctrl_subj = [f for f in sorted(os.listdir(path_to_data)) if f.startswith("ctrl_FA_")]
ctrl_subj = [int(filter(str.isdigit, ctrl_subj[j])) for j in range(len(ctrl_subj))]

ep_subj = [f for f in sorted(os.listdir(path_to_data)) if f.startswith("ep_FA_")]
ep_subj = [int(filter(str.isdigit, ep_subj[j])) for j in range(len(ep_subj))]

iugr_subj = [f for f in sorted(os.listdir(path_to_data)) if f.startswith("iugr_FA_")]
iugr_subj = [int(filter(str.isdigit, iugr_subj[j])) for j in range(len(iugr_subj))]

vav_subj = [f for f in sorted(os.listdir(path_to_data)) if f.startswith("ctrl_VAV_")]
vav_subj = [int(filter(str.isdigit, vav_subj[j])) for j in range(len(vav_subj))]

import pandas as pd
import numpy as np
import os


#########################################################
###################- Global Approach -###################
#################### --------------- ####################

# Extract global network features: deg, cc, ne for the 3 groups and concatinate them in one dataframe
dt1 = read_ctrl(ctrl_subj, control_group_filename)
dt2 = read_ep(ep_subj, ep_group_filename)
dt3 = read_iugr(iugr_subj, iugr_group_filename)
dt4 = read_vav(vav_subj, vav_group_filename)

# write data to csv
dt = pd.concat([dt1,dt2,dt3,dt4])
dt.to_csv(save_results_to + 'global_network_features.csv')


#########################################################
###################- Nodal Approach -###################
#################### --------------- ####################

# Extract global network features: deg, cc, ne for the 3 groups and concatinate them in one dataframe
dt1_nodal = read_ctrl_nodal(ctrl_subj, control_group_filename)
dt2_nodal = read_ep_nodal(ep_subj, ep_group_filename)
dt3_nodal = read_iugr_nodal(iugr_subj, iugr_group_filename)
dt4_nodal = read_vav_nodal(vav_subj, vav_group_filename)

# write data to csv
dt_nodal = pd.concat([dt1_nodal,dt2_nodal,dt3_nodal,dt4_nodal])
dt_nodal.to_csv(save_results_to + 'nodal_network_features.csv')


#########################################################
###################- Connections Approach -###################
#################### --------------- ####################

# Extract global network features: deg, cc, ne for the 3 groups and concatinate them in one dataframe
X_nd_ctrls = read_ctrl_connections(ctrl_subj, control_group_filename,number_of_brain_regions_in_connectomes)
X_nd_eps = read_ep_connections(ep_subj, ep_group_filename,number_of_brain_regions_in_connectomes)
X_nd_iugrs = read_iugr_connections(iugr_subj, iugr_group_filename,number_of_brain_regions_in_connectomes)

# Stack the data
all_mats = np.dstack((X_nd_ctrls, X_nd_eps, X_nd_iugrs))

# Get the upper triagular part of all the subjects matrices
X_connections = all_mats[np.triu_indices(all_mats.shape[0], k = 1)]

# Correct the shape
X_connections = X_connections.T

labels = np.array([1] * len(ctrl_subj) + [2] *len(ep_subj) + [3] *len(iugr_subj))
X_connections = np.insert(X_connections, X_connections.shape[1], values=labels, axis=1)

# write data to csv
np.savetxt(save_results_to + "connections_features.csv", X_connections, delimiter=",")


#########################################################
####################- Deep Approach -####################
################### fast greedy decomp ##################

# calculate the mean ctrl network and decompose it
# X_mean_ctrl = mean_ctrl(ctrl_subj, number_of_brain_regions_in_connectomes, control_group_filename)
X_mean_ctrl = mean_ctrl(vav_subj, number_of_brain_regions_in_connectomes, vav_group_filename)

# use the prior information to decompose all subj into the same number of communities
dt_ctrl_fg, clusters_fg = decompose_ctrl_fg(X_mean_ctrl, ctrl_subj, control_group_filename, verbose)
dt_ep_fg , _ = decompose_ep_fg(X_mean_ctrl, ep_subj, ep_group_filename, verbose)
dt_iugr_fg , _ = decompose_iugr_fg(X_mean_ctrl, iugr_subj, iugr_group_filename, verbose)
dt_vav_fg, _ = decompose_vav_fg(X_mean_ctrl, vav_subj, vav_group_filename, verbose)
# dt_ctrl_fg.to_csv('fast_ctrls.csv')
# dt_ep_fg.to_csv('fast_ep.csv')
# dt_iugr_fg.to_csv('fast_iugr.csv')

# write data to csv
dt_fg = pd.concat([dt_ctrl_fg, dt_ep_fg, dt_iugr_fg, dt_vav_fg], ignore_index = True)
dt_fg.to_csv(save_results_to + 'deep_fast_greedy_comm_network_features.csv')
np.save(save_results_to + "FG_clusters", clusters_fg)

#########################################################
####################- Deep Approach -####################
################### leading eigen decomp ################

dt_ctrl_lev, clusters_le = decompose_ctrl_lev(X_mean_ctrl, ctrl_subj, control_group_filename, verbose)
dt_ep_lev, _ = decompose_ep_lev(X_mean_ctrl, ep_subj, ep_group_filename, verbose)
dt_iugr_lev, _ = decompose_iugr_lev(X_mean_ctrl, iugr_subj, iugr_group_filename, verbose)
dt_vav_lev, _ = decompose_vav_lev(X_mean_ctrl, vav_subj, vav_group_filename, verbose)
# dt_ctrl_fg.to_csv('fast_ctrls.csv')
# dt_ep_fg.to_csv('fast_ep.csv')
# dt_iugr_fg.to_csv('fast_iugr.csv')

# write data to csv
dt_lev = pd.concat([dt_ctrl_lev, dt_ep_lev, dt_iugr_lev, dt_vav_lev], ignore_index = True)
dt_lev.to_csv(save_results_to + 'deep_leading_eigenvector_comm_network_features.csv')
np.save(save_results_to + "LEV_clusters", clusters_le)

# Plot the comunities of the Mean CTRL network (fast greedy algo) and save it to 'save_results_to'
plot_com, visual_style, layout = plot_communities(X_mean_ctrl)
if verbose:
	plot(plot_com, mark_groups = True, layout=layout, **visual_style)
	plot(plot_com, save_results_to + 'MEAN_FG.pdf', mark_groups = True, layout=layout, **visual_style)
else:
	plot(plot_com, save_results_to + 'MEAN_FG.pdf', mark_groups = True, layout=layout, **visual_style)


""" Until here, for Neil

os.chdir(save_results_to)


#########################################################
################## Apply LDA on Global data #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('global_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
scores_global = call_LDA_LOOCV(X, y, verbose)
print('#######################################################')
print(Fore.RED +'The accuracy for the GLOBAL approach is {}\n'.format(scores_global))
print(Style.RESET_ALL)
del(X,y)


#########################################################
################## Apply SVM on Nodal data #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('nodal_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
scores_nodal = call_SVM_LOOCV(X, y, verbose)
print('#######################################################')
print(Fore.RED +'The accuracy for the NODAL approach is {}\n'.format(scores_nodal))
print(Style.RESET_ALL)
del(X,y)

#########################################################
################## Apply SVM on Connections data #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
scores_connections = call_SVM_LOOCV(X, y, verbose)
print('#######################################################')
print(Fore.RED +'The accuracy for the Connections approach is {}\n'.format(scores_connections))
print(Style.RESET_ALL)
del(X,y)


#########################################################
################## Apply MLP on Connections data #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('connections_features.csv', header = None)
data = data.iloc[:,0:]
k=data.shape[1]-1
X, y = data.iloc[:, 0:k].values, data.iloc[:,k].values
MLP_scores_connections = call_MLP_LOOCV(X, y, verbose)
print('#######################################################')
print(Fore.RED +'The accuracy for the MLP Connections approach is {}\n'.format(MLP_scores_connections))
print(Style.RESET_ALL)
del(X,y)

#########################################################
######### Apply LDA on Deep fast greedy data ############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
data = data.iloc[:,1:]
k=data.shape[1]-1
X, y = data.iloc[0:, 0:k].values, data.iloc[0:,k].values
scores_deep_fg = call_LDA_LOOCV(X, y, verbose)
print('#######################################################')
print(Fore.RED +'The accuracy for the DEEP fast greedy approach is {}\n'.format(scores_deep_fg))
print(Style.RESET_ALL)
del(X,y)

#########################################################
###### Apply LDA on Deep leading eigen data #############
#########################################################

# Get features and target variable for all the 3 groups
data = pd.read_csv('deep_leading_eigenvector_comm_network_features.csv')
data = data.iloc[:,1:]
X, y = data.iloc[0:, 0:18].values, data.iloc[0:,18].values
scores_deep_lev = call_LDA_LOOCV(X, y, verbose)
print('#######################################################')
print(Fore.RED +'The accuracy for the DEEP leading eigen approach is {}\n'.format(scores_deep_lev))
print(Style.RESET_ALL)
del(X,y)

#########################################################
#################### Plot the results ###################
#########################################################

plot = plot_values_global_modular([scores_global, scores_deep_fg, scores_deep_lev], save_results_to, verbose)
print('#######################################################')
print(Fore.RED +'All the results have been saved in the following directory \n{}'.format(save_results_to))
print('#######################################################')
print(Style.RESET_ALL)
if verbose:
	pass
else:
	print('No results were displayed because verbose was set to {} in main.py'.format(verbose))
	print('#######################################################')

plot = plot_values_nodal_connections([scores_nodal, scores_connections], save_results_to, verbose)
print('#######################################################')
print(Fore.RED +'All the results have been saved in the following directory \n{}'.format(save_results_to))
print('#######################################################')
print(Style.RESET_ALL)
if verbose:
	pass
else:
	print('No results were displayed because verbose was set to {} in main.py'.format(verbose))
	print('#######################################################')



#########################################################
############### Get the LDA coefficients ################
#########################################################


data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')

# Get LDA coefficients for DEGREE measure
Degree_Weights_ctrl_vs_ep, Degree_Weights_ctrl_vs_iugr, Degree_Weights_ep_vs_iugr  = get_weights(data ,number_of_features = 3)

df_ep_vs_iugr_weights= pd.DataFrame({'Nodes': np.transpose(range(number_of_brain_regions_in_connectomes)) , 
						'Weights': np.transpose(np.zeros(number_of_brain_regions_in_connectomes)).reshape(number_of_brain_regions_in_connectomes,),
						'ID': np.transpose(np.zeros(number_of_brain_regions_in_connectomes)).reshape(number_of_brain_regions_in_connectomes,)})

for i in range(len(clusters_fg)):
	df_ep_vs_iugr_weights.loc[clusters_fg[i], 'Weights' ] = Degree_Weights_ep_vs_iugr[i].copy()
	df_ep_vs_iugr_weights.loc[clusters_fg[i], 'ID' ] = i + 1


# Save the results
os.chdir(path_to_data)
tmp = pd.read_csv('coordinates.node', sep=" ", header=None)
tmp.iloc[:,4] = df_ep_vs_iugr_weights['Weights'].values[:]
tmp.iloc[:,3] = df_ep_vs_iugr_weights['ID'].values[:]
tmp.to_csv(save_results_to + "EP_vs_IUGR_degree.csv", header=None, index=None, sep=' ')


## Get the names of the nodes for each module
# names = tmp.iloc[:,5]
# module_n = 3
# names[clusters_le[module_n]]


## Plot ROCs
plt = plot_ROCs(exclude_class_label=1)
plt.show()
plt.close()

plt = plot_ROCs_decomposition(exclude_class_label=1)
plt.show()
plt.close()


# Cross validated ROCs 
# use subplots_all_ROCS.py in Utilities folder inside the toolbox's main folder.

"""



