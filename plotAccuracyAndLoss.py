from plotAccuracy import *
from plotLosses import *
import matplotlib.pyplot as plt

def plotAccuracyAndLoss(file_list, figsize=(5,5), output=None, xlim=(0,100), ylim1=(0.75,1.0), ylim2=(-0.25,-0.05)):

	fig, axarr = plt.subplots(2, 1, figsize=figsize)

	for i in range(len(file_list)):
		plotAccuracy(file_list[i], fig_ax=(fig,axarr[0]), formatting=False, output="None", label=(i==0), circles=False)
		plotLosses(file_list[i], fig_ax=(fig,axarr[1]), formatting=False, output="None", label=(i==0), circles=False)

	for ax in axarr:
		ax.set_xlim(xlim)
	axarr[0].set_xticklabels([])
	axarr[0].set_ylim(ylim1)
	axarr[1].set_ylim(ylim2)

	axarr[0].set_ylabel('Accuracy', fontsize=11)
	axarr[1].set_ylabel('log(Loss)', fontsize=11)
	axarr[1].set_xlabel('Model Epoch', fontsize=11)
	axarr[1].legend(frameon=False, fontsize=11, loc='upper right')

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output, dpi=400)
		plt.close(fig)

if __name__ == '__main__':
	file_list = []
	file_list.append('./data_products/losses_addstretch_renorm_fold0_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold1_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold2_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold3_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold4_012726.pkl')
	plotAccuracyAndLoss(file_list, output='./figures/accuracy_loss_012726.pdf')
