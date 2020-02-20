"""
=========================================
Author: Serafeim Loukas, May 2018
=========================================

"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def call_LDA_LOOCV(X,y, verbose=0):
	"""
	This function applies LDA on the data and returns the LOOCV scores in 2 ways.

	Created by: Loukas Serafeim, Nov 2017

	Args:
		X: A numpy array of the input features
		y: A numpy array of the target values. Note: this shoud have shape= [n_features, ]

	Returns:
 		The mean LOOCV scores of LDA classification
	"""

	###### Standardize Data ###########
	pipe= Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())])
	loo = LeaveOneOut()
	if verbose:
		print("The number of splits is:{}\n".format(loo.get_n_splits(X)))

	########################  1st WAY ######################
	test_fold_predictions=[]
	y_test_all=[]

	for i,j in loo.split(X):

		X_train, X_test = X[i], X[j]
		y_train, y_test = y[i], y[j]
		pipe.fit(X_train, y_train)
		y_pred = pipe.predict(X_test)
		test_fold_predictions.append(y_pred)
		y_test_all.append(y_test)
	if verbose:
		print('Confusion matrix \n{}\n'.format(metrics.confusion_matrix(y_test_all, test_fold_predictions)))
		print("Accuracy is %r \n" %metrics.accuracy_score(y_test_all, test_fold_predictions))

	################ PLOT CONFUSION MATRIX PLOT #########################
	#plt.imshow(confusion_matrix(y_test_all, test_fold_predictions), interpolation='nearest', cmap=plt.cm.Blues)  
	#plt.colorbar() 
	#plt.xlabel("True label")                                                             
	#plt.ylabel("Predicted label")
	#plt.title(" The Confusion Matrix")
	### stop blocking #########3
	#plt.show(block = False)   

	###################### 2nd way using sklearn build-in functions ###################
	scores = cross_val_score(pipe, X, y, cv=loo, scoring="accuracy")
	if verbose:
		print("Accuracy of 2nd way is %r\n" %np.mean(scores))
	#plt.show()

	return np.mean(scores)


	