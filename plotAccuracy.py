import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotAccuracy(infile, output=None, figsize=(5,4), fig_ax=None, formatting=True, color_test='r', color_train='b', ls_test='-', ls_train='-', label=True, circles=True):

	with open(infile, 'rb') as f:
		D = pickle.load(f)

	test_accuracy = D['test_accuracy']
	training_accuracy = D['training_accuracy']

	if fig_ax is None:
		fig, ax = plt.subplots(1, 1, figsize=figsize)
	else:
		fig, ax = fig_ax

	x = range(1,len(training_accuracy)+1)
	if label:
		label_test = 'Validation'
		label_training = 'Training'
	else:
		label_test = None
		label_training = None

	ax.plot(x, np.array(test_accuracy)[:len(x)], color=color_test, label=label_test, ls=ls_test)
	if circles:
		ax.scatter(x, np.array(test_accuracy)[:len(x)], color=color_test, marker='o', facecolor='white', zorder=5, ls=ls_test)
	ax.plot(x, np.array(training_accuracy)[:len(x)], color=color_train, label=label_training, ls=ls_train)
	if circles:
		ax.scatter(x, np.array(training_accuracy)[:len(x)], color=color_train, marker='o', facecolor='white', zorder=5, ls=ls_train)

	if formatting:
		ax.set_xlabel('Model Epoch', fontsize=12)
		ax.set_ylabel('Accuracy', fontsize=12)
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
	plotAccuracy(infile, output='./figures/080725/accuracy.png')
	'''
	import sys
	plotAccuracy(sys.argv[1])
