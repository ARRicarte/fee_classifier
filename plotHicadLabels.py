import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def organizeLabels(folder):

	labels_unsorted = []
	times = []
	spins = []
	Rhighs = []
	folds = []
	
	files = glob.glob(os.path.join(folder, "*pkl"))
	for fold_index in range(len(files)):
		file = files[fold_index]
		print(f"Processing {file}...")
		with open(file, 'rb') as f:
			D = pickle.load(f)

		parameter_words = [name.split('/')[-1].split('.h5')[0].split('_') for name in D['names']]
		frequencies = [float(words[7][4:]) for words in parameter_words]
		labels_unsorted.extend([D['labels'][i] for i in range(len(parameter_words)) if frequencies[i] == 228.0])
		spins.extend([float(parameter_words[i][1][5:]) for i in range(len(parameter_words)) if frequencies[i] == 228.0])
		times.extend([float(parameter_words[i][2][1:]) for i in range(len(parameter_words)) if frequencies[i] == 228.0])
		Rhighs.extend([float(parameter_words[i][4][2:]) for i in range(len(parameter_words)) if frequencies[i] == 228.0])
		folds.extend([fold_index]*frequencies.count(228))

	labels_unsorted = np.array(labels_unsorted)
	times = np.array(times)
	spins = np.array(spins)
	Rhighs = np.array(Rhighs)
	folds = np.array(folds)

	print("Reorganizing...")
	#Everything is disorganized.  Compute more consistent arrays.
	Rhigh_unique = np.unique(Rhighs)
	spin_unique = np.unique(spins)
	time = np.sort(np.unique(times))
	labels = np.zeros((len(np.unique(folds)),len(spin_unique),len(Rhigh_unique),len(time)))
	time_for_sorting = np.zeros((len(np.unique(folds)),len(spin_unique),len(Rhigh_unique),len(time)))

	for fold_index in np.unique(folds):
		for R_index in range(len(Rhigh_unique)):
			for a_index in range(len(spin_unique)):
				mask = (spins == spin_unique[a_index]) & (Rhighs == Rhigh_unique[R_index]) & (folds == fold_index)
				if np.sum(mask) == 0:
					#This combination didn't exist
					continue
				time_for_sorting = times[mask]
				order = np.argsort(time_for_sorting)
				labels[fold_index,a_index,R_index,:] = labels_unsorted[mask][order]

	majority_vote = np.squeeze(np.mean(labels, axis=0))
	consensus = np.squeeze(np.all(labels>0.5, axis=0))

	outDict = {}
	outDict['labels'] = labels
	outDict['spin'] = spin_unique
	outDict['Rhigh'] = Rhigh_unique
	outDict['time'] = time
	outDict['majority_vote'] = majority_vote
	outDict['consensus'] = consensus
	return outDict

def plotHicadLabels(D, spin, Rhigh, output=None):

	a_index = np.argmin(np.abs(D['spin']-spin))
	Rh_index = np.argmin(np.abs(D['Rhigh']-Rhigh))

	fig, ax = plt.subplots(1, 1, figsize=(8,4))
	t = D['time']-D['time'][0]
	dt = np.diff(t)[0]
	t_stairs = np.concatenate([[t[0]-dt/2],t+dt/2])
	ax.stairs(D['consensus'][a_index,Rh_index,:], t_stairs, color='orange', fill=True)
	ax.plot(t, D['majority_vote'][a_index,Rh_index,:], color='dodgerblue', lw=1)

	#ax.set_xticks(np.linspace(0,1e4,11))
	ax.set_xlim(t[0], t[-1])
	ax.set_ylim(0,1)
	ax.set_xlabel(r"t - $10^5$ [$GM_\bullet/c^3$]", fontsize=11)
	ax.set_ylabel("Majority Vote", fontsize=11, color='dodgerblue')
	ax.set_title("Does this image have a flux eruption?", fontsize=11)
	ax_twin = ax.twinx()
	ax_twin.set_ylim(0,1)
	ax_twin.set_yticks([0,1])
	ax_twin.set_yticklabels(["No", "Yes"])
	ax_twin.set_ylabel("Consensus", color='orange', fontsize=11, rotation=270)

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output, dpi=300)
		plt.close(fig)

def labelAnalysis(D):

	#Compute duty cycle of cavities.

	#Compute recurrence timescale.  Need to group 1s based on proximity somehow and probably exclude singletons.

	#Compute correlation between different R_high values.
	return

if __name__ == '__main__':
	folder = './hicad_labels'
	D = organizeLabels(folder)
	plotHicadLabels(D, 0.9, 160)

	#Plot all labels
	#for a in D['spin']:
	#	for R in D['Rhigh']:
	#		plotHicadLabels(D, a, R, output=f'./figures/hicad_labels/a{a}_Rh{R}.png')
