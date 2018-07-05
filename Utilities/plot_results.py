"""
=========================================
Author: Serafeim Loukas, May 2018
=========================================

"""
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show,plot,draw
from matplotlib import style
style.use('ggplot')


def plot_values(values,save_to, verbose=0):
    """
    This functions plots the LDA results for the Global and Local frameworks.
    
    Created by: Loukas Serafeim, Nov 2017

    Args:
    	values: List of scores. e.g. [scores_global, scores_deep_fg, scores_deep_lev]
    	save_to: Directory to save the figure. e.g. '/Users/miplab/Desktop/LDA_toolbox/Results/'

    Returns:
    	Plot of the results.

    Raises:
    	Exception if the input args are not correct
    """
    
    if type(values) == list:

		scores = values
		labels = ('Global', 'Deep FG', 'Deep LE')
		x_cords1 = np.arange(len(labels))
		plt.ylim([0.5,1])
		plt.bar([0], float(scores[0]), align='center', width=0.70, color=['#E63434'],label='Global',alpha=0.65)
		plt.bar([1], float(scores[1]),align='center', width=0.70, color=['#09BA15'],label='Fast_Greddy',alpha=0.65)
		plt.bar([2], float(scores[2]),align='center', width=0.70, color=['#4D4DFF'],label='Leading_Eigenvector',alpha=0.65)
		plt.tick_params(axis='x',labelsize=12)
		plt.xticks(x_cords1, labels)
		plt.ylabel('Accuracy_LOOCV')
		plt.title('LDA classification using all the 3 Groups',fontsize=13)
		plt.legend(loc=4,prop={'size':10})
		plt.savefig(save_to + 'results_LDA.png', dpi=300)
		if verbose:
			plt.show()

    else:
        raise Exception('\n\nThe input type is not correct. Retry...\n\n')

