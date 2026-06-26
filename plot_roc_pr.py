from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import numpy as np
import pickle
import glob
import os
import prepare_files as file_prep
import matplotlib.pyplot as plt

used_labels = "./used_labels"
files = glob.glob(os.path.join(used_labels, "*.pkl"))

#Getting all the data.
fprs = []
tprs = []
precisions = []
recalls = []
thresholds1 = []
thresholds2 = []
calibration_true = []
ks = []

calibration_bin_edges = np.linspace(0,1,6)
for file in files:
	segments = file.split("_")
	k = int(segments[np.where(['fold' in thing for thing in segments])[0][0]][-1])
	ks.append(k)
	train_loader, test_loader = file_prep.generate_dataset(n_threads=1, nonfee_list='./labels/nonfee_mad_sane_weird_090925.txt', fee_list='./labels/fee_files_090925.txt', k=k)
	with open(file, 'rb') as f:
		label_dict = pickle.load(f)
	for images, labels, names in test_loader:
		labels = labels.numpy()
		#Find the labels that actually correspond to the test loader.
		inTest = np.isin(label_dict['names'], [name.split('.h5')[0]+'.h5' for name in names])
		inTest_names = np.array(label_dict['names'])[inTest]
		inTest_labels = np.array(label_dict['labels'])[inTest]

		#Map between test loader and truth labels.
		testNameToTruthIndex = {x: i for i, x in enumerate(inTest_names)}
		mapping = np.array([testNameToTruthIndex[x.split('.h5')[0]+'.h5'] for x in names])

		#Receiver Operating Characteristic
		fpr, tpr, thresholds = roc_curve(labels, inTest_labels[mapping])
		fprs.append(fpr)
		tprs.append(tpr)
		thresholds1.append(thresholds)

		#Precision, Recall
		precision, recall, thresholds = precision_recall_curve(labels, inTest_labels[mapping])
		print(len(precision), len(recall), len(thresholds))
		precisions.append(precision)
		recalls.append(recall)
		thresholds2.append(thresholds)

		#Calibration curve
		#prob_true, prob_pred = calibration_curve(labels, inTest_labels[mapping], n_bins=6, strategy='uniform')
		prob_true = []
		prob_pred = []
		for b_index in range(len(calibration_bin_edges)-1):
			inBin = (inTest_labels[mapping] >= calibration_bin_edges[b_index]) & (inTest_labels[mapping] < calibration_bin_edges[b_index+1])
			prob_true.append(np.mean(labels[inBin]))
		calibration_true.append(prob_true)

#Plotting
fig, axarr = plt.subplots(1, 3, figsize=(10,3.5))
foldToColor = dict(zip(range(5), ['b', 'r', 'g', 'purple', 'orange']))

for i in range(len(fprs)):
	axarr[0].plot(fprs[i], tprs[i], color=foldToColor[ks[i]], lw=1, alpha=0.5)
	axarr[1].plot(recalls[i], precisions[i], color=foldToColor[ks[i]], lw=1, alpha=0.5)

#Now the averages. Annoyingly, thresholds get automatically selected and I need to interpolate.
thresholds_interp = np.linspace(0,1,100)
fpr_mean = np.mean([np.interp(thresholds_interp, np.flipud(thresholds1[i]), np.flipud(fprs[i])) for i in range(5)], axis=0)
tpr_mean = np.mean([np.interp(thresholds_interp, np.flipud(thresholds1[i]), np.flipud(tprs[i])) for i in range(5)], axis=0)
recall_mean = np.mean([np.interp(thresholds_interp, np.concatenate((thresholds2[i],[1])), recalls[i]) for i in range(5)], axis=0)
precision_mean = np.mean([np.interp(thresholds_interp, np.concatenate((thresholds2[i],[1])), precisions[i]) for i in range(5)], axis=0)

axarr[0].plot(fpr_mean, tpr_mean, color='k', lw=2, zorder=5)
axarr[1].plot(recall_mean, precision_mean, color='k', lw=2, zorder=5)
axarr[0].scatter(fpr_mean[50], tpr_mean[50], marker='*', facecolor='white', edgecolor='k', s=150, zorder=10)
axarr[1].scatter(recall_mean[50], precision_mean[50], marker='*', facecolor='white', edgecolor='k', s=150, zorder=10)

print("FPR, TPR, Recall, Precision, F1:")
print(fpr_mean[50], tpr_mean[50], recall_mean[50], precision_mean[50], 2*(precision_mean[50]*recall_mean[50])/(precision_mean[50]+recall_mean[50]))
print("ROC-AUC", "PR-AUC")
print(-np.trapezoid(tpr_mean, fpr_mean), -np.trapezoid(precision_mean, recall_mean))

axarr[0].set_xlim(0,0.2)
axarr[0].set_ylim(0.7,1)
axarr[1].set_xlim((0.8,1))
axarr[1].set_ylim((0.93,1))

axarr[0].set_xlabel("FPR", fontsize=11)
axarr[0].set_ylabel("TPR", fontsize=11)
axarr[1].set_xlabel("Recall", fontsize=11)
axarr[1].set_ylabel("Precision", fontsize=11)

for i in range(len(calibration_true)):
	#ax.plot(calibration_pred[i], calibration_true[i], color=foldToColor[ks[i]], lw=1)
	axarr[2].stairs(calibration_true[i], calibration_bin_edges, color=foldToColor[ks[i]], lw=1, alpha=0.5)
axarr[2].stairs(np.mean(calibration_true, axis=0), calibration_bin_edges, color='k', lw=2, zorder=5)
axarr[2].plot([0,1], [0,1], color='grey', zorder=-1, lw=1, ls=':')
axarr[2].set_xlim(0,1)
axarr[2].set_ylim(0,1)
axarr[2].set_xticks(np.linspace(0,1,5))
axarr[2].set_yticks(np.linspace(0,1,5))
axarr[2].set_xlabel("Predicted Probability", fontsize=11)
axarr[2].set_ylabel("True Positive Fraction", fontsize=11)
fig.tight_layout()
fig.show()
