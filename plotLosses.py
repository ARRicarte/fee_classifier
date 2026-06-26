import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotLosses(infile, output=None, figsize=(5,4), log=True, fig_ax=None, formatting=True, color_test='r', color_train='b', ls_test='-', ls_train='-', label=True, circles=True):

	with open(infile, 'rb') as f:
		D = pickle.load(f)

	test_loss = np.array(D['test_loss'])
	training_loss = np.array(D['training_loss'])
	if log:
		test_loss = np.log10(test_loss)
		training_loss = np.log10(training_loss)

	if fig_ax is None:
		fig, ax = plt.subplots(1, 1, figsize=figsize)
	else:
		fig, ax = fig_ax

	x = range(1,len(training_loss)+1)
	if label:
		label_test = 'Validation'
		label_training = 'Training'
	else:
		label_test = None
		label_training = None
	ax.plot(x, test_loss[:len(x)], color=color_test, label=label_test, ls=ls_test)
	if circles:
		ax.scatter(x, test_loss[:len(x)], color=color_test, marker='o', facecolor='white', zorder=5, ls=ls_test)

	ax.plot(x, training_loss[:len(x)], color=color_train, label=label_training, ls=ls_train)
	if circles:
		ax.scatter(x, training_loss[:len(x)], color=color_train, marker='o', facecolor='white', zorder=5, ls=ls_train)

	if formatting:
		ax.set_xlabel('Model Epoch', fontsize=12)
		if log:
			ax.set_ylabel('log(Loss)', fontsize=12)
		else:
			ax.set_ylabel('Loss', fontsize=12)
		ax.set_xlim(x[0],x[-1])

		ax.legend(fontsize=12, frameon=False)
		fig.tight_layout()

	if not output in ["None", "none"]:
		if output is None:
			fig.show()
		else:
			fig.savefig(output, dpi=400)
			plt.close(fig)

if __name__ == '__main__':
	'''
	infile = './data_products/losses_sanestoo_refreeze_080725.pkl'
	plotLosses(infile, output='./figures/080725/losses.png')
	'''
	import sys
	plotLosses(sys.argv[1])
