import os
import pandas as pd
import numpy as np


### Build connectomes (mean length masked by FA) after running mrtrix preprocessing

#path to mrtrix connectome csv files
path = '/Volumes/Loukas_MIP/results_6yo_DTI/'

#path to save the results of this script
save_to = '/Volumes/Loukas_MIP/results_6yo_DTI/CONNECTOMES2/'

#get all folder names that start with "6"
subject_folders = [f for f in sorted(os.listdir(path)) if f.startswith('6') ]

#Function to check if a matrix is symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

#Create the "save_to" directory if it does not exists
if not os.path.exists(save_to):
    os.makedirs(save_to)


# Get the ID and MRI code names for the CTRL, EP, IUGR groups
IDs = pd.read_excel('/Users/miplab/Desktop/LDA_toolbox/DTI_NEW_6YO.xlsx')

ctrls_IDs = IDs['Patient ID'][0:8]
ctrls_IDs = '6' + ctrls_IDs.astype(str)

ctrls_codes = IDs['code'][0:8]
ctrls_codes = ctrls_codes.astype(str)

eps_IDs =  IDs['Patient ID'][8:31]
eps_IDs = '6' + eps_IDs.astype(str)

eps_codes = IDs['code'][8:31]
eps_codes = eps_codes.astype(str)


iugr_IDs =  IDs['Patient ID'][31:]
iugr_IDs = '6' + iugr_IDs.astype(str)

iugr_codes = IDs['code'][31:]
iugr_codes = iugr_codes.astype(str)



# Iterate over all subjects and save the results in "save_to" directory
for s in range(len(subject_folders)):

	tmp = subject_folders[s]
	tmp_path = os.path.join(path,tmp)

	con = np.array(pd.read_csv(tmp_path + '/connectome.csv', header=None, delim_whitespace=True))
	con = np.nan_to_num(con)
	W_con = np.maximum(con, con.transpose() )
	del(con)
	
	con_ml = np.array(pd.read_csv(tmp_path + '/connectome_ml.csv', header=None, delim_whitespace=True))
	con_ml = np.nan_to_num(con_ml)
	W_ml = np.maximum(con_ml, con_ml.transpose() )
	del(con_ml)

	con_fa = np.array(pd.read_csv(tmp_path + '/connectome_fa.csv', header=None, delim_whitespace=True))
	con_fa = np.nan_to_num(con_fa)
	W_fa = np.maximum(con_fa, con_fa.transpose() )
	del(con_fa)

	mask = (W_fa > 0.2) * 1.
	W_ml_fa = np.divide(W_con, W_ml)
	W_ml_fa = np.nan_to_num(W_ml_fa)
	W_ml_fa = np.multiply(W_ml_fa, mask)
	
	#zero out diagonal
	for idx in range(len(W_ml_fa)):
		W_ml_fa[idx,idx] = 0.0


	if check_symmetric(W_con) & check_symmetric(W_ml) & check_symmetric(W_fa):

		if tmp in ctrls_IDs.tolist():
			position = ctrls_IDs.tolist().index(tmp)
			name_of_current_subj = ctrls_codes.tolist()[position]
			filename_ = ''.join([l for l in name_of_current_subj if l.isdigit()]).lstrip("0")
			np.savetxt(save_to + "ctrl_FA_{}.csv".format(filename_), W_ml_fa, delimiter=",")

		if tmp in eps_IDs.tolist():
			position = eps_IDs.tolist().index(tmp)
			name_of_current_subj = eps_codes.tolist()[position]
			filename_ = ''.join([l for l in name_of_current_subj if l.isdigit()]).lstrip("0")
			np.savetxt(save_to + "ep_FA_{}.csv".format(filename_), W_ml_fa, delimiter=",")

		if tmp in iugr_IDs.tolist():
			position = iugr_IDs.tolist().index(tmp)
			name_of_current_subj = iugr_codes.tolist()[position]
			filename_ = ''.join([l for l in name_of_current_subj if l.isdigit()]).lstrip("0")
			np.savetxt(save_to + "iugr_FA_{}.csv".format(filename_), W_ml_fa, delimiter=",")


del(subject_folders, path, filename_, W_ml, W_fa, W_con, W_ml_fa, s, l, idx)



### Save for the new VAV ctrls

#path to mrtrix connectome csv files
path = '/Volumes/Loukas_MIP/results_6yo_DTI/VAV_ctrls/'
#get all folder names that start with "6"
subject_folders = [f for f in sorted(os.listdir(path)) if f.startswith('6') ]

# Iterate over all subjects and save the results in "save_to" directory
for s in range(len(subject_folders)):

	tmp = subject_folders[s]
	tmp_path = os.path.join(path,tmp)

	con = np.array(pd.read_csv(tmp_path + '/connectome.csv', header=None, delim_whitespace=True))
	con = np.nan_to_num(con)
	W_con = np.maximum(con, con.transpose() )
	del(con)
	
	con_ml = np.array(pd.read_csv(tmp_path + '/connectome_ml.csv', header=None, delim_whitespace=True))
	con_ml = np.nan_to_num(con_ml)
	W_ml = np.maximum(con_ml, con_ml.transpose() )
	del(con_ml)

	con_fa = np.array(pd.read_csv(tmp_path + '/connectome_fa.csv', header=None, delim_whitespace=True))
	con_fa = np.nan_to_num(con_fa)
	W_fa = np.maximum(con_fa, con_fa.transpose() )
	del(con_fa)

	mask = (W_fa > 0.2) * 1.
	W_ml_fa = np.divide(W_con, W_ml)
	W_ml_fa = np.nan_to_num(W_ml_fa)
	W_ml_fa = np.multiply(W_ml_fa, mask)
	
	#zero out diagonal
	for idx in range(len(W_ml_fa)):
		W_ml_fa[idx,idx] = 0.0


	if check_symmetric(W_con) & check_symmetric(W_ml) & check_symmetric(W_fa):
			filename_ = tmp.lstrip("6_")
			filename_ = ''.join([l for l in filename_ if l.isdigit()])
			np.savetxt(save_to + "ctrl_VAV_{}.csv".format(filename_), W_ml_fa, delimiter=",")


print("Done! Check the results here: {}".format(save_to))





