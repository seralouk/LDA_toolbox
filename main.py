"""
=========================================
Author: Serafeim Loukas, May 2018
=========================================

"""

import os
import sys
import cairocffi as cairo

# Define the paths to utilities, data and results directories.
path_to_utilities = '/Users/miplab/Desktop/LDA_toolbox/Utilities/'
path_to_data = '/Users/miplab/Desktop/LDA_toolbox/Dataset/'
save_results_to = '/Users/miplab/Desktop/LDA_toolbox/Results/'
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
number_of_brain_regions_in_connectomes = 83
verbose = 1 #0 do not print in console, otherwise print

# Define the finelames as they appear in "path_to_data" directory
control_group_filename = 'ctrl_FA_{}.csv'
ep_group_filename = 'ep_FA_{}.csv'
iugr_group_filename = 'iugr_FA_{}.csv'

# os.chdir(path_to_utilities)
from plot_comms import plot_communities
from decomposition_functions import *
from nodal_efficiency import nodal_eff
from LDA_functions import call_LDA_LOOCV
from plot_results import plot_values
from lda_coefficients import get_weights
# os.chdir(go_back)

if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

os.chdir(path_to_data)

num_ctrl_subs = 9 #8 ctrl subjects
num_ep_subs = 24
num_iugr_subs = 22
ctrl_subj = range(1,num_ctrl_subs) #8 ctrl subjects
ep_subj = range(1,num_ep_subs)
iugr_subj = range(1,num_iugr_subs)

#########################################################
###################- Global Approach -###################
#################### --------------- ####################

# Extract global network features: deg, cc, ne for the 3 groups and concatinate them in one dataframe
dt1 = read_ctrl(ctrl_subj, control_group_filename)
dt2 = read_ep(ep_subj, ep_group_filename)
dt3 = read_iugr(iugr_subj, iugr_group_filename)

# write data to csv
dt = pd.concat([dt1,dt2,dt3])
dt.to_csv(save_results_to + 'global_network_features.csv')

#########################################################
####################- Deep Approach -####################
################### fast greedy decomp ##################

# calculate the mean ctrl network and decompose it
X_mean_ctrl = mean_ctrl(ctrl_subj, number_of_brain_regions_in_connectomes, control_group_filename)

# use the prior information to decompose all subj into the same number of communities
dt_ctrl_fg, clusters_fg = decompose_ctrl_fg(X_mean_ctrl, ctrl_subj, control_group_filename, verbose)
dt_ep_fg = decompose_ep_fg(X_mean_ctrl, ep_subj, ep_group_filename, verbose)
dt_iugr_fg = decompose_iugr_fg(X_mean_ctrl, iugr_subj, iugr_group_filename, verbose)
# dt_ctrl_fg.to_csv('fast_ctrls.csv')
# dt_ep_fg.to_csv('fast_ep.csv')
# dt_iugr_fg.to_csv('fast_iugr.csv')

# write data to csv
dt_fg = pd.concat([dt_ctrl_fg, dt_ep_fg, dt_iugr_fg], ignore_index = True)
dt_fg.to_csv(save_results_to + 'deep_fast_greedy_comm_network_features.csv')

#########################################################
####################- Deep Approach -####################
################### leading eigen decomp ################

dt_ctrl_lev, clusters_le = decompose_ctrl_lev(X_mean_ctrl, ctrl_subj, control_group_filename, verbose)
dt_ep_lev = decompose_ep_lev(X_mean_ctrl, ep_subj, ep_group_filename, verbose)
dt_iugr_lev = decompose_iugr_lev(X_mean_ctrl, iugr_subj, iugr_group_filename, verbose)
# dt_ctrl_fg.to_csv('fast_ctrls.csv')
# dt_ep_fg.to_csv('fast_ep.csv')
# dt_iugr_fg.to_csv('fast_iugr.csv')

# write data to csv
dt_lev = pd.concat([dt_ctrl_lev, dt_ep_lev, dt_iugr_lev], ignore_index = True)
dt_lev.to_csv(save_results_to + 'deep_leading_eigenvector_comm_network_features.csv')

# Plot the comunities of the Mean CTRL network (fast greedy algo) and save it to 'save_results_to'
plot_com, visual_style, layout = plot_communities(X_mean_ctrl)
if verbose:
	plot(plot_com, mark_groups = True, layout=layout, **visual_style)
	plot(plot_com, save_results_to + 'graph.pdf', mark_groups = True, layout=layout, **visual_style)
else:
	plot(plot_com, save_results_to + 'graph.pdf', mark_groups = True, layout=layout, **visual_style)

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
print('The accuracy for the Global approach is {}\n'.format(scores_global))
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
print('The accuracy for the Deep fast greedy approach is {}\n'.format(scores_deep_fg))
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
print('The accuracy for the Deep leading eigen approach is {}\n'.format(scores_deep_lev))
del(X,y)

#########################################################
#################### Plot the results ###################
#########################################################

plot = plot_values([scores_global, scores_deep_fg, scores_deep_lev], save_results_to, verbose)
print('#######################################################')
print('All the results have been saved in the following directory \n{}'.format(save_results_to))
print('#######################################################')
if verbose:
	pass
else:
	print('No results were displayed because verbose was set to {} in main.py'.format(verbose))
	print('#######################################################')

#########################################################
############### Get the LDA coefficients ################
#########################################################
data = pd.read_csv('deep_fast_greedy_comm_network_features.csv')
# Get LDA coefficients for degree measure
Degree_Weights_ctrl_vs_ep, Degree_Weights_ctrl_vs_iugr, Degree_Weights_ep_vs_iugr  = get_weights(data ,number_of_features = 3)

df_ep_vs_iugr_weights= pd.DataFrame({'Nodes': np.transpose(range(number_of_brain_regions_in_connectomes)) , 
						'Weights': np.transpose(np.zeros(number_of_brain_regions_in_connectomes)).reshape(number_of_brain_regions_in_connectomes,),
						'ID': np.transpose(np.zeros(number_of_brain_regions_in_connectomes)).reshape(number_of_brain_regions_in_connectomes,)})

for i in range(len(clusters)):
	df_ep_vs_iugr_weights.loc[clusters[i], 'Weights' ] = Degree_Weights_ep_vs_iugr[i]
	df_ep_vs_iugr_weights.loc[clusters[i], 'ID' ] = i + 1

os.chdir(path_to_data)
# Save the results
tmp = pd.read_csv('clustering1.node', sep=" ", header=None)
tmp.iloc[:,4] = df_ep_vs_iugr_weights['Weights'].values
tmp.iloc[:,3] = df_ep_vs_iugr_weights['ID'].values
tmp.to_csv(save_results_to + "EP_vs_IUGR_degree.csv", header=None, index=None, sep=' ')



# Get the names of the nodes for each module

# names = tmp.iloc[:,5]
# module_n = 3
# names[clusters_le[module_n]]





